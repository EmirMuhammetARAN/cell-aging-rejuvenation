"""
LDM v13 Fine-Tuning: Continuous Biological Age Conditioning.
Fine-tunes LDM v12 using continuous senescence scores (Picard lifespan scale P1-P40)
from Classifier v3 (ConvNeXt-Tiny). Enables exact continuous control (e.g. +10 passages, %30 aged).
"""
import os
import sys
import gc
import re
import glob
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v13 import CellLDM_v13
from models.ldm.dataset_v13 import LDMDatasetV13

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

EXPERIMENT_NAME = "v13_continuous_age"
BASE_CHECKPOINT = "checkpoint_v12_v4_data_lpips_last.pt"

if __name__ == '__main__':
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 25
    WARMUP_EPOCHS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 2e-5
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5
    SAVE_INTERVAL = 5

    CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', EXPERIMENT_NAME)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print(f"\n{'='*70}")
    print(f"       INITIALIZING LDM v13 (CONTINUOUS BIOLOGICAL AGE CONDITIONAL)   ")
    print(f"{'='*70}")

    model = CellLDM_v13(latent_loss_weight=0.05)
    model.to(DEVICE, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema()

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.unet.parameters()),
        lr=LR, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.05, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    # Resume or Load Base v12 Model
    last_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_last.pt')
    base_ckpt_path = os.path.join(CHECKPOINT_DIR, BASE_CHECKPOINT)

    start_epoch = 0
    best_loss = float('inf')

    if os.path.exists(last_ckpt_path):
        print(f"Resuming v13 training from: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location='cpu')
        model.unet.load_state_dict(ckpt['unet_state_dict'])
        if 'ema_unet_state_dict' in ckpt:
            model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    elif os.path.exists(base_ckpt_path):
        print(f"Bootstrapping LDM v13 from base v12 model: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location='cpu')
        sd = ckpt['unet_state_dict']
        if 'class_embedding.weight' in sd:
            sd['class_embedding.orig_embed.weight'] = sd.pop('class_embedding.weight')
        model.unet.load_state_dict(sd, strict=False)
        if 'ema_unet_state_dict' in ckpt:
            ema_sd = ckpt['ema_unet_state_dict']
            if 'class_embedding.weight' in ema_sd:
                ema_sd['class_embedding.orig_embed.weight'] = ema_sd.pop('class_embedding.weight')
            model.ema_unet.load_state_dict(ema_sd, strict=False)
        print("Pretrained v12 weights loaded successfully into v13 architecture!")
    else:
        print(f"Base checkpoint not found at {base_ckpt_path}! Starting from scratch.")

    # Datasets
    train_dataset = LDMDatasetV13(root_dir=root_dir, split='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    # Exemplar young cell for trajectory tracking
    test_young_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    young_files = sorted([f for f in os.listdir(test_young_dir) if f.endswith(('.jpg', '.png'))])
    fixed_young_img = None
    if young_files:
        p_exemplar = os.path.join(test_young_dir, young_files[0])
        from PIL import Image
        fixed_young_img = val_transform(Image.open(p_exemplar).convert('RGB')).unsqueeze(0).to(DEVICE)

    print(f"\nStarting LDM v13 fine-tune for {NUM_EPOCHS} epochs on {DEVICE}...")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.unet.train()
        epoch_loss = 0.0
        steps = 0
        optimizer.zero_grad()
        accum = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [v13 Fine-Tune]")
        for imgs, age_scores in pbar:
            imgs = imgs.to(DEVICE, memory_format=torch.channels_last)
            age_scores = age_scores.to(DEVICE)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16 if USE_BFLOAT16 else torch.float32):
                loss = model(imgs, continuous_ages=age_scores)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()
            accum += 1

            if accum % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                model.update_ema()
                optimizer.zero_grad()

            loss_val = loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_loss += loss_val
            steps += 1

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        lr_scheduler.step()
        avg_loss = epoch_loss / steps
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} Complete | Average Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Validation & Visual Trajectory Generation
        if (epoch + 1) % VAL_FREQ == 0 or (epoch + 1) == NUM_EPOCHS:
            model.unet.eval()
            print(f"Generating validation biological age trajectory at epoch {epoch+1}...")
            
            # 1. Unconditional generation across 7 passages
            passages_to_sample = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
            sampled_frames = []
            for p in passages_to_sample:
                frame = model.sample_passage(passage=p, num_samples=1, device=DEVICE, use_ema=True, guidance_scale=3.0, num_steps=40)
                sampled_frames.append(frame.cpu())
            grid_sample = torch.cat(sampled_frames, dim=0)
            save_image(grid_sample, os.path.join(RESULTS_DIR, f'sample_trajectory_epoch_{epoch+1}.png'), nrow=len(passages_to_sample))

            # 2. Translation trajectory on fixed test young cell
            if fixed_young_img is not None:
                trajectories = [fixed_young_img.cpu() * 0.5 + 0.5] # Baseline original
                for delta in [+5.0, +10.0, +15.0, +20.0, +25.0]:
                    trans = model.translate_continuous(
                        fixed_young_img, delta_passage=delta, current_passage=12.0,
                        strength=0.65, num_steps=40, use_ema=True, guidance_scale=3.0
                    )
                    trajectories.append(trans.cpu())
                grid_trans = torch.cat(trajectories, dim=0)
                save_image(grid_trans, os.path.join(RESULTS_DIR, f'aging_trajectory_epoch_{epoch+1}.png'), nrow=len(trajectories))
                print(f"  -> Saved trajectory figures to {RESULTS_DIR}")

            # Save checkpoint
            ckpt_data = {
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_loss
            }
            torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_last.pt'))
            if (epoch + 1) % SAVE_INTERVAL == 0:
                torch.save(ckpt_data, os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch+1}.pt'))
                print(f"  -> Saved milestone checkpoint at epoch {epoch+1}")

    print("\n[DONE] LDM v13 continuous biological age fine-tuning complete!")
