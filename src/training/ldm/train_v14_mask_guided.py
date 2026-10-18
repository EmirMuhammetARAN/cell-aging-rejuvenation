"""
LDM v14 Training: Mask-Guided / Region-Focused Fine-Tuning.
===========================================================
Fine-tunes LDM v12 (checkpoint_v12_v4_data_lpips_last.pt) using spatial mask-weighted loss:
  - Young (Class 0): 24 px dilation envelope (accommodates hypertrophic growth)
  - Senescent (Class 1): 8 px dilation envelope (accommodates cellular contraction)
  - Loss Objective: Mask-Weighted Diffusion Noise MSE + Latent Structural L1 (w=0.1)
  - Background Stabilizer Weight: 0.05
  - Learning Rate: 1e-5 with 2-epoch warmup and 28-epoch cosine annealing (Total 30 epochs)
  - Validation: Evaluated with Low-FID regime (Aging: str=0.75, cfg=4.0 | Rejuv: str=0.65, cfg=3.5)
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import gc
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

os.environ["PYTHONIOENCODING"] = "utf-8"

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4294967296"

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v14 import CellLDM_v14
from models.ldm.dataset_v14 import LDMDatasetV14

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

EXPERIMENT_NAME = "v14_mask_guided"
BASE_CHECKPOINT = "checkpoint_v12_v4_data_lpips_last.pt"

if __name__ == '__main__':
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 4
    NUM_EPOCHS = 30
    WARMUP_EPOCHS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    LR = 1e-5
    USE_BFLOAT16 = True
    MAX_GRAD_NORM = 1.0
    VAL_FREQ = 5
    SAVE_INTERVAL = 5
    LPIPS_WEIGHT = 0.1
    BG_WEIGHT = 0.05

    CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', EXPERIMENT_NAME)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*76}")
    print(f"       INITIALIZING LDM v14 (MASK-GUIDED REGION-FOCUSED FINE-TUNING)       ")
    print(f"{'='*76}")
    print(f"  - Base Model:         {BASE_CHECKPOINT}")
    print(f"  - Target Epochs:      {NUM_EPOCHS} (Warmup: {WARMUP_EPOCHS})")
    print(f"  - Learning Rate:      {LR}")
    print(f"  - Loss Formulation:   Mask-Weighted MSE + Latent L1 ({LPIPS_WEIGHT})")
    print(f"  - Dilation Radii:     Aging = 24 px, Rejuvenation = 8 px")
    print(f"  - Background Weight:  {BG_WEIGHT} (Cell region = 1.0)")
    print(f"  - Validation Regime:  Low-FID (Aging: s=0.75 cfg=4.0 | Rejuv: s=0.65 cfg=3.5)")
    print(f"{'='*76}\n")

    # Datasets & Loaders
    train_dataset = LDMDatasetV14(root_dir, split='train', young_dilation=24, senes_dilation=8)
    val_dataset = LDMDatasetV14(root_dir, split='test', young_dilation=24, senes_dilation=8)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False
    )

    # Initialize Model v14
    model = CellLDM_v14(lpips_weight=LPIPS_WEIGHT, bg_weight=BG_WEIGHT)
    model.to(DEVICE, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.unet.parameters()),
        lr=LR, weight_decay=1e-2, betas=(0.9, 0.999), eps=1e-8
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-7
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[WARMUP_EPOCHS]
    )

    # Resume or Load Base v12 Weights
    last_ckpt_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_last.pt')
    base_ckpt_path = os.path.join(CHECKPOINT_DIR, BASE_CHECKPOINT)

    start_epoch = 0
    best_val_loss = float('inf')

    if os.path.exists(last_ckpt_path):
        print(f"[*] Resuming v14 training from: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location='cpu')
        model.unet.load_state_dict(ckpt['unet_state_dict'])
        if 'ema_unet_state_dict' in ckpt:
            model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in ckpt:
            lr_scheduler.load_state_dict(ckpt['lr_scheduler_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_path = os.path.join(CHECKPOINT_DIR, f'best_model_{EXPERIMENT_NAME}.pt')
        if os.path.exists(best_path):
            try:
                b_ckpt = torch.load(best_path, map_location='cpu')
                if 'best_val_loss' in b_ckpt:
                    best_val_loss = min(best_val_loss, b_ckpt['best_val_loss'])
                del b_ckpt
            except Exception:
                pass
        print(f"    Resumed from epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
        del ckpt
        gc.collect()
    elif os.path.exists(base_ckpt_path):
        print(f"[*] Initializing v14 weights from v12 base: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location='cpu')
        for key in ['unet_state_dict', 'ema_unet_state_dict']:
            if key in ckpt and 'class_embedding.weight' in ckpt[key]:
                w = ckpt[key]['class_embedding.weight']
                if w.shape[0] == 2:
                    pad = torch.zeros(1, w.shape[1], device=w.device)
                    ckpt[key]['class_embedding.weight'] = torch.cat([w, pad], dim=0)
        model.unet.load_state_dict(ckpt['unet_state_dict'])
        if 'ema_unet_state_dict' in ckpt:
            model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
        print("    Base v12 weights loaded successfully.")
        del ckpt
        gc.collect()
    else:
        print(f"[!] Warning: Base checkpoint {base_ckpt_path} not found. Starting uninitialized.")

    amp_dtype = torch.bfloat16 if (USE_BFLOAT16 and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"\n[*] Training began at Epoch {start_epoch + 1}/{NUM_EPOCHS}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.unet.train()
        total_loss = 0.0
        num_batches = 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for step, (images, labels, masks) in enumerate(pbar):
            images = images.to(DEVICE, memory_format=torch.channels_last)
            labels = labels.to(DEVICE)
            masks = masks.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = model(images, labels=labels, masks=masks)
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            loss.backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.unet.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_ema()

            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            cur_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f"{total_loss / num_batches:.4f}", 'lr': f"{cur_lr:.2e}"})

        lr_scheduler.step()
        avg_train_loss = total_loss / max(1, num_batches)

        # Validation & Exemplar Generation
        if (epoch + 1) % VAL_FREQ == 0 or (epoch + 1) == NUM_EPOCHS:
            model.unet.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for v_img, v_lbl, v_msk in val_loader:
                    v_img = v_img.to(DEVICE, memory_format=torch.channels_last)
                    v_lbl = v_lbl.to(DEVICE)
                    v_msk = v_msk.to(DEVICE)
                    with torch.amp.autocast('cuda', dtype=amp_dtype):
                        v_l = model(v_img, labels=v_lbl, masks=v_msk)
                    val_loss += v_l.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(1, val_batches)
            print(f"\n[Epoch {epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {cur_lr:.2e}")

            # Generate validation translation samples with Low-FID parameters
            try:
                young_idx = next((i for i, lbl in enumerate(val_dataset.labels) if lbl == 0), None)
                senes_idx = next((i for i, lbl in enumerate(val_dataset.labels) if lbl == 1), None)
                if young_idx is not None and senes_idx is not None:
                    y_sample = val_dataset[young_idx][0].unsqueeze(0).to(DEVICE)
                    s_sample = val_dataset[senes_idx][0].unsqueeze(0).to(DEVICE)

                    # Low-FID Aging translation: strength=0.75, CFG=4.0
                    target_senes = torch.ones(1, dtype=torch.long, device=DEVICE)
                    trans_senes = model.translate(y_sample, target_senes, strength=0.75, guidance_scale=4.0, num_steps=50)

                    # Low-FID Rejuvenation translation: strength=0.65, CFG=3.5
                    target_young = torch.zeros(1, dtype=torch.long, device=DEVICE)
                    trans_young = model.translate(s_sample, target_young, strength=0.65, guidance_scale=3.5, num_steps=50)

                    y_vis = (y_sample.clamp(-1, 1) + 1) / 2
                    s_vis = (s_sample.clamp(-1, 1) + 1) / 2
                    grid_y2s = torch.cat([y_vis.cpu(), trans_senes.cpu()], dim=0)
                    grid_s2y = torch.cat([s_vis.cpu(), trans_young.cpu()], dim=0)

                    save_image(grid_y2s, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}_low_fid_aging.png'), nrow=2)
                    save_image(grid_s2y, os.path.join(RESULTS_DIR, f'epoch_{epoch+1}_low_fid_rejuvenation.png'), nrow=2)
                    print(f"  [+] Saved validation Low-FID translation exemplars to {RESULTS_DIR}")
            except Exception as e:
                print(f"  [!] Exemplar generation note: {e}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_path = os.path.join(CHECKPOINT_DIR, f'best_model_{EXPERIMENT_NAME}.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'best_val_loss': best_val_loss,
                    'unet_state_dict': model.unet.state_dict(),
                    'ema_unet_state_dict': model.ema_unet.state_dict(),
                }, best_path)
                print(f"  [*] Best model saved (Val Loss: {best_val_loss:.4f}) -> {best_path}")

        # Regular milestone checkpoint
        if (epoch + 1) % SAVE_INTERVAL == 0:
            milestone_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'unet_state_dict': model.unet.state_dict(),
                'ema_unet_state_dict': model.ema_unet.state_dict(),
            }, milestone_path)
            print(f"  [+] Milestone checkpoint saved: {milestone_path}")

        # Always update last.pt for uninterrupted resume
        torch.save({
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'unet_state_dict': model.unet.state_dict(),
            'ema_unet_state_dict': model.ema_unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        }, last_ckpt_path)

    print(f"\n{'='*76}")
    print(f"[DONE] LDM v14 Mask-Guided Fine-Tuning Completed Successfully!")
    print(f"  Final weights: {last_ckpt_path}")
    print(f"{'='*76}\n")
