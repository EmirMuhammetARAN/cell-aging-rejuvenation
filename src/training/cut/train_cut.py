"""
CUT Training: Train two independent CUT models for bidirectional translation.
  Model A: Young -> Senescent (Aging)
  Model B: Senescent -> Young (Rejuvenation)
"""
import os, sys, torch, random, gc
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.cut.cut_model import CUTModel

# ============================================
# CONFIG
# ============================================
DEVICE = 'cuda'
EPOCHS = 200
BATCH_SIZE = 1
LR = 2e-4
BETA1, BETA2 = 0.5, 0.999
SAVE_EVERY = 10
LAMBDA_NCE = 1.0
LAMBDA_IDT = 1.0
RESUME_EPOCH = 90  # Set >0 to resume from a specific checkpoint

DATA_YOUNG = os.path.join(root_dir, 'data', 'processed_v2', 'train', 'young')
DATA_SENES = os.path.join(root_dir, 'data', 'processed_v2', 'train', 'senescent')
CKPT_DIR = os.path.join(root_dir, 'checkpoints', 'cut')
SAMPLE_DIR = os.path.join(root_dir, 'results', 'cut')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


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


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = UnpairedDataset(DATA_YOUNG, DATA_SENES, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=0, pin_memory=True, drop_last=True)

print(f"Dataset: {len(dataset)} pairs (young={dataset.len_a}, senescent={dataset.len_b})")
print(f"Training for {EPOCHS} epochs with PatchNCE contrastive loss")

# ============================================
# MODELS: Two CUT models for bidirectional
# ============================================
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

# LR schedulers (linear decay after epoch 100)
def get_scheduler(optimizer):
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - 100) / 100.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

sched_G_aging = get_scheduler(opt_G_aging)
sched_D_aging = get_scheduler(opt_D_aging)
sched_G_reju = get_scheduler(opt_G_reju)
sched_D_reju = get_scheduler(opt_D_reju)

# Load fixed test images for visual tracking
test_young_dir = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
test_senes_dir = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')
test_young_files = sorted([f for f in os.listdir(test_young_dir) if f.startswith('young_')])[:1]
test_senes_files = sorted(os.listdir(test_senes_dir))[:1]

fixed_young = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(
    transforms.ToTensor()(Image.open(os.path.join(test_young_dir, test_young_files[0])).convert('RGB'))
).unsqueeze(0).to(DEVICE) if test_young_files else None

fixed_senes = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(
    transforms.ToTensor()(Image.open(os.path.join(test_senes_dir, test_senes_files[0])).convert('RGB'))
).unsqueeze(0).to(DEVICE) if test_senes_files else None

# ============================================
# RESUME FROM CHECKPOINT
# ============================================
if RESUME_EPOCH > 0:
    ckpt_path = os.path.join(CKPT_DIR, f'cut_epoch_{RESUME_EPOCH}.pth')
    if os.path.exists(ckpt_path):
        print(f"\nLoading checkpoint from epoch {RESUME_EPOCH}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model_aging.G.load_state_dict(ckpt['aging_G'])
        model_aging.D.load_state_dict(ckpt['aging_D'])
        model_aging.mlp_heads.load_state_dict(ckpt['aging_mlp'])
        
        model_reju.G.load_state_dict(ckpt['reju_G'])
        model_reju.D.load_state_dict(ckpt['reju_D'])
        model_reju.mlp_heads.load_state_dict(ckpt['reju_mlp'])
        
        # Advance schedulers
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

# ============================================
# TRAINING LOOP
# ============================================
best_G_loss = float('inf')

for epoch in range(RESUME_EPOCH + 1, EPOCHS + 1):
    model_aging.train()
    model_reju.train()

    epoch_losses = {'G_aging': 0, 'D_aging': 0, 'G_reju': 0, 'D_reju': 0,
                    'NCE_aging': 0, 'NCE_reju': 0}

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
    for i, (young, senes) in enumerate(pbar):
        young = young.to(DEVICE)
        senes = senes.to(DEVICE)

        # === AGING MODEL (Young -> Senescent) ===
        losses_a = model_aging.train_step(young, senes)

        opt_D_aging.zero_grad()
        opt_G_aging.zero_grad()

        losses_a['loss_G'].backward()
        losses_a['loss_D'].backward()

        opt_G_aging.step()
        opt_D_aging.step()

        # === REJUVENATION MODEL (Senescent -> Young) ===
        losses_r = model_reju.train_step(senes, young)

        opt_D_reju.zero_grad()
        opt_G_reju.zero_grad()

        losses_r['loss_G'].backward()
        losses_r['loss_D'].backward()

        opt_G_reju.step()
        opt_D_reju.step()

        epoch_losses['G_aging'] += losses_a['loss_G'].item()
        epoch_losses['D_aging'] += losses_a['loss_D'].item()
        epoch_losses['NCE_aging'] += losses_a['loss_G_nce'].item()
        epoch_losses['G_reju'] += losses_r['loss_G'].item()
        epoch_losses['D_reju'] += losses_r['loss_D'].item()
        epoch_losses['NCE_reju'] += losses_r['loss_G_nce'].item()

        pbar.set_postfix({
            'G_a': f"{losses_a['loss_G'].item():.3f}",
            'D_a': f"{losses_a['loss_D'].item():.3f}",
            'NCE_a': f"{losses_a['loss_G_nce'].item():.3f}",
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

    # Save samples
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

        # Save checkpoints
        torch.save({
            'epoch': epoch,
            'aging_G': model_aging.G.state_dict(),
            'aging_D': model_aging.D.state_dict(),
            'aging_mlp': model_aging.mlp_heads.state_dict(),
            'reju_G': model_reju.G.state_dict(),
            'reju_D': model_reju.D.state_dict(),
            'reju_mlp': model_reju.mlp_heads.state_dict(),
        }, os.path.join(CKPT_DIR, f'cut_epoch_{epoch}.pth'))
        print(f"  -> Saved checkpoint & samples (epoch {epoch})")

print("\n✅ Training complete!")
