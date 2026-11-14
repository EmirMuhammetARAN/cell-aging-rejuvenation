"""
CUT Training: Train two independent CUT models for bidirectional translation on processed_v5.
  Model A: Young -> Senescent (Aging)
  Model B: Senescent -> Young (Rejuvenation)
Optimized with AMP mixed precision for NVIDIA RTX 4080 GPU.
"""
import os, sys, csv, time, torch, random, gc
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from models.cut.cut_model import CUTModel

class UnpairedDataset(Dataset):
    def __init__(self, dir_a, dir_b, transform=None):
        self.files_a = sorted([os.path.join(dir_a, f) for f in os.listdir(dir_a) if f.endswith(('.jpg', '.png'))])
        self.files_b = sorted([os.path.join(dir_b, f) for f in os.listdir(dir_b) if f.endswith(('.jpg', '.png'))])
        self.transform = transform
        self.len_a = len(self.files_a)
        self.len_b = len(self.files_b)

    def __len__(self):
        return max(self.len_a, self.len_b)

    def __getitem__(self, idx):
        img_a = Image.open(self.files_a[idx % self.len_a]).convert('RGB')
        img_b = Image.open(self.files_b[random.randint(0, self.len_b - 1)]).convert('RGB')
        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b


def get_scheduler(optimizer):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - 100) / 100.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    SEED = 2026
    set_seed(SEED)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 200
    BATCH_SIZE = 1
    LR = 2e-4
    BETA1, BETA2 = 0.5, 0.999
    SAVE_EVERY = 5
    LAMBDA_NCE = 1.0
    LAMBDA_IDT = 1.0
    RESUME_EPOCH = -1  # Set -1 to auto-detect latest checkpoint, 0 for scratch, or >0 for specific

    DATA_YOUNG = os.path.join(root_dir, 'data', 'processed_v5', 'train', 'young')
    DATA_SENES = os.path.join(root_dir, 'data', 'processed_v5', 'train', 'senescent')
    CKPT_DIR = os.path.join(root_dir, 'checkpoints', 'cut')
    SAMPLE_DIR = os.path.join(root_dir, 'results', 'cut')
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=90),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = UnpairedDataset(DATA_YOUNG, DATA_SENES, transform)
    dl_gen = torch.Generator().manual_seed(SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True,
                            drop_last=True, worker_init_fn=seed_worker, generator=dl_gen)

    print(f"Dataset (processed_v5): {len(dataset)} pairs (young={dataset.len_a}, senescent={dataset.len_b})")
    print(f"Training for {EPOCHS} epochs with PatchNCE contrastive loss on {DEVICE}")

    # Models: Two CUT models for bidirectional translation
    # Model A: Young -> Senescent (Aging)
    model_aging = CUTModel(device=DEVICE, lambda_nce=LAMBDA_NCE, lambda_idt=LAMBDA_IDT).to(DEVICE)
    opt_G_aging = torch.optim.Adam(
        list(model_aging.G.parameters()) + list(model_aging.mlp_heads.parameters()),
        lr=LR, betas=(BETA1, BETA2))
    opt_D_aging = torch.optim.Adam(model_aging.D.parameters(), lr=LR, betas=(BETA1, BETA2))

    # Model B: Senescent -> Young (Rejuvenation)
    model_reju = CUTModel(device=DEVICE, lambda_nce=LAMBDA_NCE, lambda_idt=LAMBDA_IDT).to(DEVICE)
    opt_G_reju = torch.optim.Adam(
        list(model_reju.G.parameters()) + list(model_reju.mlp_heads.parameters()),
        lr=LR, betas=(BETA1, BETA2))
    opt_D_reju = torch.optim.Adam(model_reju.D.parameters(), lr=LR, betas=(BETA1, BETA2))

    sched_G_aging = get_scheduler(opt_G_aging)
    sched_D_aging = get_scheduler(opt_D_aging)
    sched_G_reju = get_scheduler(opt_G_reju)
    sched_D_reju = get_scheduler(opt_D_reju)

    # Load fixed validation images for visual tracking (zero test leakage)
    test_young_dir = os.path.join(root_dir, 'data', 'processed_v5', 'val', 'young')
    test_senes_dir = os.path.join(root_dir, 'data', 'processed_v5', 'val', 'senescent')
    test_young_files = sorted([f for f in os.listdir(test_young_dir) if f.endswith(('.jpg', '.png'))])[:1]
    test_senes_files = sorted([f for f in os.listdir(test_senes_dir) if f.endswith(('.jpg', '.png'))])[:1]

    fixed_young = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(
        transforms.ToTensor()(Image.open(os.path.join(test_young_dir, test_young_files[0])).convert('RGB'))
    ).unsqueeze(0).to(DEVICE) if test_young_files else None

    fixed_senes = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(
        transforms.ToTensor()(Image.open(os.path.join(test_senes_dir, test_senes_files[0])).convert('RGB'))
    ).unsqueeze(0).to(DEVICE) if test_senes_files else None

    scaler = torch.amp.GradScaler('cuda')

    # RESUME FROM CHECKPOINT
    last_ckpt_path = os.path.join(CKPT_DIR, 'cut_last.pth')
    ckpt_path = None
    best_G_loss = float('inf')

    if RESUME_EPOCH == -1:
        candidates = {}
        for f in os.listdir(CKPT_DIR):
            if f.startswith('cut_epoch_') and f.endswith('.pth'):
                try:
                    ep_num = int(f.replace('cut_epoch_', '').replace('.pth', ''))
                    candidates[ep_num] = os.path.join(CKPT_DIR, f)
                except ValueError:
                    pass
        if os.path.exists(last_ckpt_path):
            try:
                last_meta = torch.load(last_ckpt_path, map_location='cpu')
                last_ep = int(last_meta.get('epoch', 0))
                if last_ep > 0:
                    candidates[last_ep] = last_ckpt_path
                del last_meta
            except Exception as e:
                print(f"[WARN] Could not inspect cut_last.pth: {e}")

        if candidates:
            RESUME_EPOCH = max(candidates.keys())
            ckpt_path = candidates[RESUME_EPOCH]
        else:
            RESUME_EPOCH = 0
    elif RESUME_EPOCH > 0:
        ckpt_path = os.path.join(CKPT_DIR, f'cut_epoch_{RESUME_EPOCH}.pth')

    if RESUME_EPOCH > 0 and ckpt_path and os.path.exists(ckpt_path):
        print(f"\nLoading checkpoint from {os.path.basename(ckpt_path)} (epoch {RESUME_EPOCH})...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model_aging.G.load_state_dict(ckpt['aging_G'])
        model_aging.D.load_state_dict(ckpt['aging_D'])
        model_aging.mlp_heads.load_state_dict(ckpt['aging_mlp'])

        model_reju.G.load_state_dict(ckpt['reju_G'])
        model_reju.D.load_state_dict(ckpt['reju_D'])
        model_reju.mlp_heads.load_state_dict(ckpt['reju_mlp'])

        if 'opt_G_aging' in ckpt:
            opt_G_aging.load_state_dict(ckpt['opt_G_aging'])
            opt_D_aging.load_state_dict(ckpt['opt_D_aging'])
            opt_G_reju.load_state_dict(ckpt['opt_G_reju'])
            opt_D_reju.load_state_dict(ckpt['opt_D_reju'])
        if 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        best_G_loss = ckpt.get('best_G_loss', best_G_loss)

        for _ in range(RESUME_EPOCH):
            sched_G_aging.step()
            sched_D_aging.step()
            sched_G_reju.step()
            sched_D_reju.step()
        del ckpt
        print(f"Resumed successfully. Starting from epoch {RESUME_EPOCH + 1}\n")
    else:
        if RESUME_EPOCH > 0:
            print(f"\nCheckpoint for epoch {RESUME_EPOCH} not found! Starting from scratch.\n")
        RESUME_EPOCH = 0

    log_path = os.path.join(SAMPLE_DIR, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'G_aging', 'D_aging', 'NCE_aging', 'G_reju', 'D_reju', 'NCE_reju', 'lr', 'epoch_time_s'])

    for epoch in range(RESUME_EPOCH + 1, EPOCHS + 1):
        t0 = time.time()
        model_aging.train()
        model_reju.train()

        epoch_losses = {'G_aging': 0.0, 'D_aging': 0.0, 'G_reju': 0.0, 'D_reju': 0.0,
                        'NCE_aging': 0.0, 'NCE_reju': 0.0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}", mininterval=15.0)
        for i, (young, senes) in enumerate(pbar):
            young = young.to(DEVICE, memory_format=torch.channels_last)
            senes = senes.to(DEVICE, memory_format=torch.channels_last)

            # Aging model: Young -> Senescent
            set_requires_grad(model_aging.D, True)
            opt_D_aging.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                fake_senes = model_aging.G(young)
                loss_D_real_a = model_aging.compute_gan_loss(model_aging.D(senes), True)
                loss_D_fake_a = model_aging.compute_gan_loss(model_aging.D(fake_senes.detach()), False)
                loss_D_a = 0.5 * (loss_D_real_a + loss_D_fake_a)
            scaler.scale(loss_D_a).backward()
            scaler.step(opt_D_aging)

            set_requires_grad(model_aging.D, False)
            opt_G_aging.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred_fake_a = model_aging.D(fake_senes)
                loss_G_gan_a = model_aging.compute_gan_loss(pred_fake_a, True)
                loss_G_nce_a = model_aging.compute_nce_loss(young, fake_senes) * model_aging.lambda_nce
                if model_aging.lambda_idt > 0:
                    idt_senes = model_aging.G(senes)
                    loss_G_idt_a = model_aging.compute_nce_loss(senes, idt_senes) * model_aging.lambda_idt
                else:
                    loss_G_idt_a = torch.tensor(0.0, device=DEVICE)
                loss_G_a = loss_G_gan_a + loss_G_nce_a + loss_G_idt_a
            scaler.scale(loss_G_a).backward()
            scaler.step(opt_G_aging)

            # Rejuvenation model: Senescent -> Young
            set_requires_grad(model_reju.D, True)
            opt_D_reju.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                fake_young = model_reju.G(senes)
                loss_D_real_r = model_reju.compute_gan_loss(model_reju.D(young), True)
                loss_D_fake_r = model_reju.compute_gan_loss(model_reju.D(fake_young.detach()), False)
                loss_D_r = 0.5 * (loss_D_real_r + loss_D_fake_r)
            scaler.scale(loss_D_r).backward()
            scaler.step(opt_D_reju)

            set_requires_grad(model_reju.D, False)
            opt_G_reju.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                pred_fake_r = model_reju.D(fake_young)
                loss_G_gan_r = model_reju.compute_gan_loss(pred_fake_r, True)
                loss_G_nce_r = model_reju.compute_nce_loss(senes, fake_young) * model_reju.lambda_nce
                if model_reju.lambda_idt > 0:
                    idt_young = model_reju.G(young)
                    loss_G_idt_r = model_reju.compute_nce_loss(young, idt_young) * model_reju.lambda_idt
                else:
                    loss_G_idt_r = torch.tensor(0.0, device=DEVICE)
                loss_G_r = loss_G_gan_r + loss_G_nce_r + loss_G_idt_r
            scaler.scale(loss_G_r).backward()
            scaler.step(opt_G_reju)

            scaler.update()

            epoch_losses['G_aging'] += loss_G_a.item()
            epoch_losses['D_aging'] += loss_D_a.item()
            epoch_losses['NCE_aging'] += loss_G_nce_a.item()
            epoch_losses['G_reju'] += loss_G_r.item()
            epoch_losses['D_reju'] += loss_D_r.item()
            epoch_losses['NCE_reju'] += loss_G_nce_r.item()

            pbar.set_postfix({
                'Ga': f"{loss_G_a.item():.2f}",
                'Da': f"{loss_D_a.item():.2f}",
                'Gr': f"{loss_G_r.item():.2f}",
                'Dr': f"{loss_D_r.item():.2f}",
            })

        n = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n

        sched_G_aging.step()
        sched_D_aging.step()
        sched_G_reju.step()
        sched_D_reju.step()

        ep_time = time.time() - t0
        lr_now = opt_G_aging.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"G_aging:{epoch_losses['G_aging']:.4f} D_aging:{epoch_losses['D_aging']:.4f} "
              f"NCE_aging:{epoch_losses['NCE_aging']:.4f} | "
              f"G_reju:{epoch_losses['G_reju']:.4f} D_reju:{epoch_losses['D_reju']:.4f} "
              f"NCE_reju:{epoch_losses['NCE_reju']:.4f} | LR:{lr_now:.6f} ({ep_time:.1f}s)")
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch,
                f"{epoch_losses['G_aging']:.5f}", f"{epoch_losses['D_aging']:.5f}", f"{epoch_losses['NCE_aging']:.5f}",
                f"{epoch_losses['G_reju']:.5f}", f"{epoch_losses['D_reju']:.5f}", f"{epoch_losses['NCE_reju']:.5f}",
                f"{lr_now:.6e}", f"{ep_time:.1f}",
            ])

        if epoch % SAVE_EVERY == 0:
            model_aging.eval()
            model_reju.eval()
            with torch.no_grad():
                if fixed_young is not None:
                    fake_old = model_aging.G(fixed_young)
                    save_image(fake_old * 0.5 + 0.5,
                               os.path.join(SAMPLE_DIR, f'aging_epoch_{epoch}.png'))
                if fixed_senes is not None:
                    fake_young = model_reju.G(fixed_senes)
                    save_image(fake_young * 0.5 + 0.5,
                               os.path.join(SAMPLE_DIR, f'reju_epoch_{epoch}.png'))

            avg_G_both = (epoch_losses['G_aging'] + epoch_losses['G_reju']) / 2.0
            is_best = avg_G_both < best_G_loss
            if is_best:
                best_G_loss = avg_G_both

            compact_state = {
                'epoch': epoch,
                'aging_G': model_aging.G.state_dict(),
                'aging_D': model_aging.D.state_dict(),
                'aging_mlp': model_aging.mlp_heads.state_dict(),
                'reju_G': model_reju.G.state_dict(),
                'reju_D': model_reju.D.state_dict(),
                'reju_mlp': model_reju.mlp_heads.state_dict(),
                'best_G_loss': best_G_loss,
                'epoch_losses': epoch_losses,
            }

            # Full resumable state (single rolling file to avoid filling Drive)
            full_resume_state = {
                **compact_state,
                'opt_G_aging': opt_G_aging.state_dict(),
                'opt_D_aging': opt_D_aging.state_dict(),
                'opt_G_reju': opt_G_reju.state_dict(),
                'opt_D_reju': opt_D_reju.state_dict(),
                'scaler': scaler.state_dict(),
            }
            torch.save(full_resume_state, last_ckpt_path)

            if epoch % 10 == 0:
                torch.save(compact_state, os.path.join(CKPT_DIR, f'cut_epoch_{epoch}.pth'))
                print(f"  -> Saved compact milestone cut_epoch_{epoch}.pth & cut_last.pth")
            else:
                print(f"  -> Saved rolling resume checkpoint cut_last.pth & samples (epoch {epoch})")

            if is_best:
                torch.save(compact_state, os.path.join(CKPT_DIR, 'cut_best.pth'))
                print(f"  -> [BEST] New best CUT model saved! Avg G loss: {best_G_loss:.4f}")

    print("\n[DONE] CUT Training complete!")
