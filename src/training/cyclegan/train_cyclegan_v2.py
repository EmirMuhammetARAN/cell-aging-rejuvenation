"""CycleGAN v2 Training: processed_v2 + augmentation + LPIPS cycle consistency."""
import os
import sys
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

if __name__ == "__main__":
    # ===== CONFIG =====
    YOUNG_PATH = os.path.join(root_dir, 'data', 'processed_v2', 'train', 'young')
    SENESCENT_PATH = os.path.join(root_dir, 'data', 'processed_v2', 'train', 'senescent')
    BATCH_SIZE = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_EPOCHS = 200
    CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'cyclegan_v2')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    START_EPOCH = 0
    RESUME_EPOCH = 0
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, f'cyclegan_v2_epoch_{RESUME_EPOCH}.pth') if RESUME_EPOCH > 0 else None

    # ===== DATA =====
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
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, persistent_workers=True)

    print(f"Dataset: {len(dataset)} pairs (young={len(dataset.young_files)}, senescent={len(dataset.senescent_files)})")
    print(f"Training for {NUM_EPOCHS} epochs with LPIPS cycle consistency + augmentation")

    # ===== MODEL =====
    model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE, use_lpips=False, lpips_weight=0.0)
    model.to(DEVICE, memory_format=torch.channels_last)

    best_loss_G = float('inf')

    if RESUME_PATH and os.path.isfile(RESUME_PATH):
        model.load_state_dict(torch.load(RESUME_PATH, map_location=DEVICE))
        print(f'Checkpoint loaded from epoch {RESUME_EPOCH}.')
        START_EPOCH = RESUME_EPOCH
    else:
        print('Starting from scratch.')

    model = torch.compile(model)

    # LR scheduler: linear decay after NUM_EPOCHS // 2
    def lr_lambda(epoch):
        if epoch < NUM_EPOCHS // 2:
            return 1.0
        else:
            return 1.0 - (epoch - NUM_EPOCHS // 2) / (NUM_EPOCHS // 2)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(model._orig_mod.optimizer_G, lr_lambda=lr_lambda)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(model._orig_mod.optimizer_D_A, lr_lambda=lr_lambda)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(model._orig_mod.optimizer_D_B, lr_lambda=lr_lambda)

    # Step schedulers to correct position if resuming
    for _ in range(START_EPOCH):
        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

    scaler = torch.amp.GradScaler('cuda')

    # ===== TRAINING LOOP =====
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        epoch_loss_G = []
        epoch_loss_D_A = []
        epoch_loss_D_B = []

        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
        for batch in pbar:
            loss_G, loss_D_A, loss_D_B = model.train_step(batch, scaler)
            epoch_loss_G.append(loss_G.item())
            epoch_loss_D_A.append(loss_D_A.item())
            epoch_loss_D_B.append(loss_D_B.item())
            pbar.set_postfix({
                'G': f'{loss_G:.3f}',
                'D_A': f'{loss_D_A:.3f}',
                'D_B': f'{loss_D_B:.3f}'
            })

        avg_G = sum(epoch_loss_G) / len(epoch_loss_G)
        avg_DA = sum(epoch_loss_D_A) / len(epoch_loss_D_A)
        avg_DB = sum(epoch_loss_D_B) / len(epoch_loss_D_B)
        
        lr_current = scheduler_G.get_last_lr()[0]
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] G:{avg_G:.4f} D_A:{avg_DA:.4f} D_B:{avg_DB:.4f} LR:{lr_current:.6f}')

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # Save best
        if avg_G < best_loss_G:
            best_loss_G = avg_G
            torch.save(model._orig_mod.state_dict(), os.path.join(CHECKPOINT_DIR, 'cyclegan_best_v2.pth'))
            print(f'  -> New best model! G loss: {best_loss_G:.4f}')

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model._orig_mod.state_dict(), os.path.join(CHECKPOINT_DIR, f'cyclegan_v2_epoch_{epoch+1}.pth'))
            print(f'  -> Checkpoint saved for epoch {epoch+1}')

        # Sample every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_young_dir = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
            test_senes_dir = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')

            # Young -> Senescent
            test_img = Image.open(os.path.join(test_young_dir, os.listdir(test_young_dir)[0])).convert('RGB')
            test_tensor = transform_test(test_img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_senes = model._orig_mod.G_AB(test_tensor)
            fake_img = (fake_senes.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            transforms.ToPILImage()(fake_img).save(os.path.join(RESULTS_DIR, f'aging_epoch_{epoch+1}.png'))

            # Senescent -> Young
            test_img2 = Image.open(os.path.join(test_senes_dir, os.listdir(test_senes_dir)[0])).convert('RGB')
            test_tensor2 = transform_test(test_img2).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    fake_young = model._orig_mod.G_BA(test_tensor2)
            fake_img2 = (fake_young.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
            transforms.ToPILImage()(fake_img2).save(os.path.join(RESULTS_DIR, f'reju_epoch_{epoch+1}.png'))

            model.train()

    print(f"\nTraining complete! Best G loss: {best_loss_G:.4f}")
