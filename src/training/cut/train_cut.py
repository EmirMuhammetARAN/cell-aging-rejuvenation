"""
CUT Training: Train two independent CUT models for bidirectional translation on processed_v4.
  Model A: Young -> Senescent (Aging)
  Model B: Senescent -> Young (Rejuvenation)
Optimized with AMP mixed precision for NVIDIA RTX 4080 GPU.
"""
import os, sys, torch, random, gc
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

# ============================================
# DATASET DEFINITION
# ============================================
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


# ============================================
# MAIN TRAINING PIPELINE
# ============================================
if __name__ == '__main__':
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 200
    BATCH_SIZE = 1
    LR = 2e-4
    BETA1, BETA2 = 0.5, 0.999
    SAVE_EVERY = 10
    LAMBDA_NCE = 1.0
    LAMBDA_IDT = 1.0
    RESUME_EPOCH = -1  # Set -1 to auto-detect latest checkpoint, 0 for scratch, or >0 for specific

    DATA_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'train', 'young')
    DATA_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'train', 'senescent')
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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True, persistent_workers=True, drop_last=True)

    print(f"Dataset (processed_v4): {len(dataset)} pairs (young={dataset.len_a}, senescent={dataset.len_b})")
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

    # Load fixed test images for visual tracking
    test_young_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    test_senes_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
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
    if RESUME_EPOCH == -1:
        existing_ckpts = [f for f in os.listdir(CKPT_DIR) if f.startswith('cut_epoch_') and f.endswith('.pth')]
        if existing_ckpts:
            epochs_found = [int(f.split('_')[-1].split('.')[0]) for f in existing_ckpts]
            RESUME_EPOCH = max(epochs_found)
        else:
            RESUME_EPOCH = 0

    if RESUME_EPOCH > 0:
        ckpt_path = os.path.join(CKPT_DIR, f'cut_epoch_{RESUME_EPOCH}.pth')
        if os.path.exists(ckpt_path):
            print(f"\nLoading checkpoint from epoch {RESUME_EPOCH}...")
            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            model_aging.G.load_state_dict(ckpt['aging_G'])
            model_aging.D.load_state_dict(ckpt['aging_D'])
            model_aging.mlp_heads.load_state_dict(ckpt['aging_mlp'])
            
            model_reju.G.load_state_dict(ckpt['reju_G'])
            model_reju.D.load_state_dict(ckpt['reju_D'])
            model_reju.mlp_heads.load_state_dict(ckpt['reju_mlp'])
            
            for _ in range(RESUME_EPOCH):
                sched_G_aging.step()
                sched_D_aging.step()
                sched_G_reju.step()
                sched_D_reju.step()
            print(f"Resumed successfully. Starting from epoch {RESUME_EPOCH + 1}\n")
        else:
            print(f"\nCheckpoint cut_epoch_{RESUME_EPOCH}.pth not found! Starting from scratch.\n")
            RESUME_EPOCH = 0
    else:
        RESUME_EPOCH = 0

    best_G_loss = float('inf')

    # TRAINING LOOP
    for epoch in range(RESUME_EPOCH + 1, EPOCHS + 1):
        model_aging.train()
        model_reju.train()

        epoch_losses = {'G_aging': 0.0, 'D_aging': 0.0, 'G_reju': 0.0, 'D_reju': 0.0,
                        'NCE_aging': 0.0, 'NCE_reju': 0.0}

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (young, senes) in enumerate(pbar):
            young = young.to(DEVICE, memory_format=torch.channels_last)
            senes = senes.to(DEVICE, memory_format=torch.channels_last)

            # ------------------------------------------------------------------
            # 1. AGING MODEL: Young -> Senescent
            # ------------------------------------------------------------------
            # Train Discriminator D
            set_requires_grad(model_aging.D, True)
            opt_D_aging.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                fake_senes = model_aging.G(young)
                loss_D_real_a = model_aging.compute_gan_loss(model_aging.D(senes), True)
                loss_D_fake_a = model_aging.compute_gan_loss(model_aging.D(fake_senes.detach()), False)
                loss_D_a = 0.5 * (loss_D_real_a + loss_D_fake_a)
            scaler.scale(loss_D_a).backward()
            scaler.step(opt_D_aging)

            # Train Generator G and PatchMLP heads
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

            # ------------------------------------------------------------------
            # 2. REJUVENATION MODEL: Senescent -> Young
            # ------------------------------------------------------------------
            # Train Discriminator D
            set_requires_grad(model_reju.D, True)
            opt_D_reju.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                fake_young = model_reju.G(senes)
                loss_D_real_r = model_reju.compute_gan_loss(model_reju.D(young), True)
                loss_D_fake_r = model_reju.compute_gan_loss(model_reju.D(fake_young.detach()), False)
                loss_D_r = 0.5 * (loss_D_real_r + loss_D_fake_r)
            scaler.scale(loss_D_r).backward()
            scaler.step(opt_D_reju)

            # Train Generator G and PatchMLP heads
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

        # Average losses
        n = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= n

        # Schedulers
        sched_G_aging.step()
        sched_D_aging.step()
        sched_G_reju.step()
        sched_D_reju.step()

        lr_now = opt_G_aging.param_groups[0]['lr']
        print(f"Epoch [{epoch}/{EPOCHS}] "
              f"G_aging:{epoch_losses['G_aging']:.4f} D_aging:{epoch_losses['D_aging']:.4f} "
              f"NCE_aging:{epoch_losses['NCE_aging']:.4f} | "
              f"G_reju:{epoch_losses['G_reju']:.4f} D_reju:{epoch_losses['D_reju']:.4f} "
              f"NCE_reju:{epoch_losses['NCE_reju']:.4f} | LR:{lr_now:.6f}")

        # Save samples & checkpoints
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

            # Save periodic checkpoint
            ckpt_state = {
                'epoch': epoch,
                'aging_G': model_aging.G.state_dict(),
                'aging_D': model_aging.D.state_dict(),
                'aging_mlp': model_aging.mlp_heads.state_dict(),
                'reju_G': model_reju.G.state_dict(),
                'reju_D': model_reju.D.state_dict(),
                'reju_mlp': model_reju.mlp_heads.state_dict(),
                'opt_G_aging': opt_G_aging.state_dict(),
                'opt_D_aging': opt_D_aging.state_dict(),
                'opt_G_reju': opt_G_reju.state_dict(),
                'opt_D_reju': opt_D_reju.state_dict(),
                'epoch_losses': epoch_losses,
            }
            torch.save(ckpt_state, os.path.join(CKPT_DIR, f'cut_epoch_{epoch}.pth'))
            print(f"  -> Saved periodic checkpoint & samples (epoch {epoch})")

            # Save best model
            avg_G_both = (epoch_losses['G_aging'] + epoch_losses['G_reju']) / 2.0
            if avg_G_both < best_G_loss:
                best_G_loss = avg_G_both
                torch.save(ckpt_state, os.path.join(CKPT_DIR, 'cut_best.pth'))
                print(f"  -> [BEST] New best CUT model saved! Avg G loss: {best_G_loss:.4f}")

    print("\n[DONE] CUT Training complete!")
