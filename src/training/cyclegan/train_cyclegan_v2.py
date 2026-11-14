"""CycleGAN v2 Training on processed_v5 (MSE GAN + L1 Cycle + Identity, geometric augmentation)."""
import csv
import importlib.util
import os
import random
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from src.data.dataset_maker import DatasetMaker
from models.cyclegan.generator_resnet import GeneratorResNet
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.cyclegan_gan_model import CycleGANModel

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')


def set_seed(seed=2026):
    """Set deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """Seed DataLoader worker processes deterministically."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    SEED = 2026
    set_seed(SEED)

    YOUNG_PATH = os.path.join(root_dir, 'data', 'processed_v5', 'train', 'young')
    SENESCENT_PATH = os.path.join(root_dir, 'data', 'processed_v5', 'train', 'senescent')
    BATCH_SIZE = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 200
    CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints', 'cyclegan')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'cyclegan_v2_v5')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    START_EPOCH = 0
    RESUME_EPOCH = -1  # -1 = auto-detect latest epoch checkpoint; 0 = from scratch
    if RESUME_EPOCH == -1:
        found = [
            int(f.replace('cyclegan_v2_epoch_', '').replace('.pth', ''))
            for f in os.listdir(CHECKPOINT_DIR)
            if f.startswith('cyclegan_v2_epoch_') and f.endswith('.pth')
        ]
        RESUME_EPOCH = max(found) if found else 0
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, f'cyclegan_v2_epoch_{RESUME_EPOCH}.pth') if RESUME_EPOCH > 0 else None

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=90),
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = DatasetMaker(young_path=YOUNG_PATH, senescent_path=SENESCENT_PATH, transform=transform_train)
    dl_gen = torch.Generator().manual_seed(SEED)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=dl_gen,
    )

    print(f"Dataset: {len(dataset)} pairs (young={len(dataset.young_files)}, senescent={len(dataset.senescent_files)})")
    print(f"Training CycleGAN v2 for {NUM_EPOCHS} epochs (seed={SEED}, MSE+Cycle+Idt, geometric aug)")

    base_model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE, use_lpips=False, lpips_weight=0.0)
    base_model.to(DEVICE, memory_format=torch.channels_last)

    best_loss_G = float('inf')

    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        checkpoint = torch.load(RESUME_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            base_model.load_state_dict(checkpoint['model_state'])
            if 'optimizer_G' in checkpoint:
                base_model.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
                base_model.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
                base_model.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
            best_loss_G = checkpoint.get('best_loss_G', float('inf'))
            print(f"Checkpoint loaded from epoch {RESUME_EPOCH}. Restored best G loss: {best_loss_G:.4f}")
        else:
            base_model.load_state_dict(checkpoint)
            print(f"Checkpoint loaded from epoch {RESUME_EPOCH}. (Old format, best loss reset)")
        START_EPOCH = RESUME_EPOCH
    else:
        print('Starting from scratch.')

    use_compile = importlib.util.find_spec("triton") is not None
    if use_compile:
        print("Triton detected: compiling CycleGANModel with torch.compile")
        model = torch.compile(base_model)
    else:
        print("Triton not available (e.g. Windows): running eager CycleGANModel")
        model = base_model

    # LR scheduler: linear decay after NUM_EPOCHS // 2
    def lr_lambda(epoch):
        if epoch < NUM_EPOCHS // 2:
            return 1.0
        else:
            return 1.0 - (epoch - NUM_EPOCHS // 2) / (NUM_EPOCHS // 2)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(base_model.optimizer_G, lr_lambda=lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(base_model.optimizer_D_A, lr_lambda=lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(base_model.optimizer_D_B, lr_lambda=lr_lambda)

    # Step schedulers to correct position if resuming
    for _ in range(START_EPOCH):
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

    scaler = torch.amp.GradScaler('cuda')
    log_path = os.path.join(RESULTS_DIR, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'loss_G', 'loss_D_A', 'loss_D_B', 'lr', 'epoch_time_s'])

    for epoch in range(START_EPOCH, NUM_EPOCHS):
        t0 = time.time()
        epoch_loss_G = []
        epoch_loss_D_A = []
        epoch_loss_D_B = []

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}', mininterval=15.0)
        for batch in pbar:
            loss_G, loss_D_A, loss_D_B = base_model.train_step(batch, scaler) if not use_compile else model.train_step(batch, scaler)
            epoch_loss_G.append(loss_G.item())
            epoch_loss_D_A.append(loss_D_A.item())
            epoch_loss_D_B.append(loss_D_B.item())
            pbar.set_postfix({
                'G': f'{loss_G:.3f}',
                'D_A': f'{loss_D_A:.3f}',
                'D_B': f'{loss_D_B:.3f}'
            })

        ep_time = time.time() - t0
        avg_G = sum(epoch_loss_G) / len(epoch_loss_G)
        avg_DA = sum(epoch_loss_D_A) / len(epoch_loss_D_A)
        avg_DB = sum(epoch_loss_D_B) / len(epoch_loss_D_B)
        
        lr_current = scheduler_G.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] G:{avg_G:.4f} D_A:{avg_DA:.4f} D_B:{avg_DB:.4f} LR:{lr_current:.6f} ({ep_time:.1f}s)')
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch + 1, f'{avg_G:.5f}', f'{avg_DA:.5f}', f'{avg_DB:.5f}', f'{lr_current:.6e}', f'{ep_time:.1f}'])

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Save best
        if avg_G < best_loss_G:
            best_loss_G = avg_G
            state = {
                'epoch': epoch + 1,
                'model_state': base_model.state_dict(),
                'optimizer_G': base_model.optimizer_G.state_dict(),
                'optimizer_D_A': base_model.optimizer_D_A.state_dict(),
                'optimizer_D_B': base_model.optimizer_D_B.state_dict(),
                'best_loss_G': best_loss_G
            }
            torch.save(state, os.path.join(CHECKPOINT_DIR, 'cyclegan_best_v2_resumed.pth'))
            print(f'  -> New best model! G loss: {best_loss_G:.4f}')

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            state = {
                'epoch': epoch + 1,
                'model_state': base_model.state_dict(),
                'optimizer_G': base_model.optimizer_G.state_dict(),
                'optimizer_D_A': base_model.optimizer_D_A.state_dict(),
                'optimizer_D_B': base_model.optimizer_D_B.state_dict(),
                'best_loss_G': best_loss_G
            }
            torch.save(state, os.path.join(CHECKPOINT_DIR, f'cyclegan_v2_epoch_{epoch+1}.pth'))
            print(f'  -> Checkpoint saved for epoch {epoch+1}')

        # Sample every 10 epochs (val split only, zero test leakage)
        if (epoch + 1) % 10 == 0:
            base_model.eval()
            val_young_dir = os.path.join(root_dir, 'data', 'processed_v5', 'val', 'young')
            val_senes_dir = os.path.join(root_dir, 'data', 'processed_v5', 'val', 'senescent')
            val_y_file = sorted(f for f in os.listdir(val_young_dir) if f.lower().endswith(('.jpg', '.png')))[0]
            val_s_file = sorted(f for f in os.listdir(val_senes_dir) if f.lower().endswith(('.jpg', '.png')))[0]

            # Young -> Senescent
            test_img = Image.open(os.path.join(val_young_dir, val_y_file)).convert('RGB')
            test_tensor = transform_test(test_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_senes = base_model.G_AB(test_tensor)
            fake_img = (fake_senes.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
            transforms.ToPILImage()(fake_img).save(os.path.join(RESULTS_DIR, f'aging_epoch_{epoch+1}.png'))

            # Senescent -> Young
            test_img2 = Image.open(os.path.join(val_senes_dir, val_s_file)).convert('RGB')
            test_tensor2 = transform_test(test_img2).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_young = base_model.G_BA(test_tensor2)
            fake_img2 = (fake_young.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
            transforms.ToPILImage()(fake_img2).save(os.path.join(RESULTS_DIR, f'reju_epoch_{epoch+1}.png'))

            base_model.train()

    print(f"\nTraining complete! Best G loss: {best_loss_G:.4f}")
