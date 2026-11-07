"""
LDM v16 - Stage 2: MSC Fine-Tuning with Continuous Pseudo-Passages
===================================================================
  - Base Model:   checkpoints/ldm/best_model_v16_fibroblast_passage.pt (Stage 1 pre-trained)
  - Data:         8,312 MSC crops (data/processed_v4/train) conditioned on continuous pseudo-passages [2.0, 38.5]
                  computed via Classifier v3 (ConvNeXt-Tiny) on the Picard lifespan scale.
  - Conditioning: cell_type = 1 (MSC), continuous pseudo-passage (sinusoidal + MLP)
  - Validation:   645 MSC crops (data/processed_v4/test) evaluated on EMA weights.
  - Optimization: AdamW, peak LR (5e-5 -> 1e-6 cosine decay), warmup 2 epochs, bf16 autocast, EMA.
                  Supports seamless Cosine Warm-Restart for extended schedules (e.g., 30 -> 90 epochs).
  - Seed:         2026
  - Outputs:      checkpoints/ldm/checkpoint_<exp>_last.pt, best_model_<exp>.pt,
                  results/generated/ldm/<exp>/ (translation sweeps), train_log.csv

Usage:
    python src/training/ldm/train_v16_stage2_msc.py                 # full run (auto-resumes or warm-restarts)
    python src/training/ldm/train_v16_stage2_msc.py --smoke 20      # sanity / pipeline test
"""

import argparse
import csv
import gc
import math
import os
import random
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v16 import CellLDM_v16
from models.ldm.dataset_v16 import MSCPassageDataset, CELL_MSC

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

MSC_SWEEP_PASSAGES = [3.5, 6.0, 8.5, 12.0, 15.0, 18.0]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', default='v16_stage2_msc')
    p.add_argument('--base_ckpt', default='best_model_v16_fibroblast_passage.pt',
                   help='Name or path of pre-trained Stage 1 checkpoint in checkpoints/ldm')
    p.add_argument('--epochs', type=int, default=150,
                   help='Total Stage 2 epochs (continues from 120 -> 150 with contiguous calibration and no 90-deg rotation)')
    p.add_argument('--warmup_epochs', type=float, default=2.0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=4e-5,
                   help='Peak learning rate for Stage 2 MSC optimization')
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--lpips_weight', type=float, default=0.1)
    p.add_argument('--cfg_drop', type=float, default=0.10)
    p.add_argument('--sampler_power', type=float, default=0.0,
                   help='Intra-class inverse frequency power (0.0 = pure 50/50 class balance with uniform per-cell weight)')
    p.add_argument('--sampler_bin', type=float, default=1.0,
                   help='Passage bin width for intra-class passage sampling')
    p.add_argument('--val_stride', type=int, default=1, help='use every N-th val image for fast val loss (1 = full 616-cell val set)')
    p.add_argument('--val_freq', type=int, default=2)
    p.add_argument('--save_interval', type=int, default=15,
                   help='Save milestone checkpoints every N epochs (e.g., 45, 60, 75, 90, 105, 120, 135, 150)')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--smoke', type=int, default=0,
                   help='run only N micro-batches + one val pass + one sweep, then exit')
    return p.parse_args()


@torch.no_grad()
def run_validation(model, val_loader, device, amp_dtype):
    """Deterministic val epsilon-MSE on EMA weights across young (P<=6.0), intermediate (6.0<P<12.0), and senescent (P>=12.0)."""
    model.unet.eval()
    model.cond.eval()
    sums = {'all': 0.0, 'young': 0.0, 'inter': 0.0, 'senes': 0.0}
    cnts = {k: 0 for k in sums}
    with torch.random.fork_rng(devices=[0] if device == 'cuda' else []):
        torch.manual_seed(1234)
        for img, pas, ct in val_loader:
            img = img.to(device, memory_format=torch.channels_last)
            pas, ct = pas.to(device), ct.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                l = model(img, pas, ct, per_sample=True, use_ema=True)
            l = l.float().cpu().numpy()
            p = pas.cpu().numpy()
            for g, m in (('all', np.ones_like(p, bool)),
                         ('young', p <= 6.0),
                         ('inter', (p > 6.0) & (p < 12.0)),
                         ('senes', p >= 12.0)):
                sums[g] += float(l[m].sum())
                cnts[g] += int(m.sum())
    return {k: (sums[k] / cnts[k] if cnts[k] else float('nan')) for k in sums}


@torch.no_grad()
def save_sweep(model, val_ds, results_dir, tag, device):
    """Generates noise->image MSC sweep and SDEdit Aging & Rejuvenation sweeps on real test MSC crops."""
    # 1. Noise -> Image sweep with cell_type = 1 (MSC)
    g = torch.Generator(device=device).manual_seed(42)
    noise = torch.randn(1, 4, 64, 64, device=device, generator=g).repeat(len(MSC_SWEEP_PASSAGES), 1, 1, 1)
    ct_msc = torch.full((len(MSC_SWEEP_PASSAGES),), CELL_MSC, dtype=torch.long, device=device)
    imgs = model.sample(MSC_SWEEP_PASSAGES, cell_type=ct_msc, noise=noise, guidance_scale=3.5, num_steps=50)
    save_image(imgs.cpu(), os.path.join(results_dir, f'{tag}_msc_noise2img.png'), nrow=len(MSC_SWEEP_PASSAGES))

    # 2. SDEdit Aging: Real Young MSC -> translated to older passages
    young_idx = int(np.argmin(val_ds.passages))
    real_y = val_ds[young_idx][0].unsqueeze(0).to(device)
    real_y_vis = (real_y.clamp(-1, 1) + 1) / 2
    rows_aging = [real_y_vis.cpu()]
    aging_targets = [6.0, 8.5, 12.0, 15.0, 18.0]
    for tp in aging_targets:
        out = model.translate(real_y, tp, cell_type=CELL_MSC, strength=0.65, guidance_scale=3.5, num_steps=50)
        rows_aging.append(out.cpu())
    save_image(torch.cat(rows_aging, 0), os.path.join(results_dir, f'{tag}_sdedit_aging.png'), nrow=len(rows_aging))

    # 3. SDEdit Rejuvenation: Real Senescent MSC -> translated to younger passages
    senes_idx = int(np.argmax(val_ds.passages))
    real_s = val_ds[senes_idx][0].unsqueeze(0).to(device)
    real_s_vis = (real_s.clamp(-1, 1) + 1) / 2
    rows_rejuv = [real_s_vis.cpu()]
    rejuv_targets = [15.0, 12.0, 8.5, 6.0, 3.5]
    for tp in rejuv_targets:
        out = model.translate(real_s, tp, cell_type=CELL_MSC, strength=0.65, guidance_scale=3.5, num_steps=50)
        rows_rejuv.append(out.cpu())
    save_image(torch.cat(rows_rejuv, 0), os.path.join(results_dir, f'{tag}_sdedit_rejuvenation.png'), nrow=len(rows_rejuv))


def main():
    args = parse_args()
    smoke = args.smoke > 0
    exp = 'v16_stage2_smoke' if smoke else args.exp_name

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    CKPT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', exp)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds = MSCPassageDataset(root_dir, 'train')
    val_ds = MSCPassageDataset(root_dir, 'val')
    val_subset = torch.utils.data.Subset(val_ds, list(range(0, len(val_ds), args.val_stride)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_ds.make_sampler(args.seed, power=args.sampler_power, bin_size=args.sampler_bin),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(2, args.num_workers), pin_memory=True)

    steps_per_epoch = len(train_loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    print(f"\n{'='*78}\n   LDM v16 STAGE 2 | MSC FINE-TUNING (PSEUDO-PASSAGES) | exp={exp}\n{'='*78}")
    print(f"  MSC crops train/val: {len(train_ds)}/{len(val_ds)} (val subset {len(val_subset)})")
    print(f"  batch {args.batch_size} x accum {args.accum} = eff {args.batch_size*args.accum} | "
          f"{steps_per_epoch} opt-steps/epoch | {total_steps} total | warmup {warmup_steps}")
    print(f"  lr {args.lr} -> {args.min_lr} | wd {args.weight_decay} | lpips_w {args.lpips_weight} | "
          f"cfg_drop {args.cfg_drop} | sampler(bin={args.sampler_bin}, pow={args.sampler_power}) | "
          f"amp {amp_dtype} | seed {args.seed}\n{'='*78}\n")

    model = CellLDM_v16(cfg_drop_prob=args.cfg_drop, lpips_weight=args.lpips_weight)
    model.to(DEVICE)
    model.unet.to(memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema()

    # Load Stage 1 Pretrained Checkpoint
    base_ckpt_path = args.base_ckpt
    if not os.path.isabs(base_ckpt_path):
        base_ckpt_path = os.path.join(CKPT_DIR, base_ckpt_path)

    if os.path.exists(base_ckpt_path):
        print(f"[*] Loading Stage 1 pre-trained weights from: {base_ckpt_path}", flush=True)
        ckpt_stage1 = torch.load(base_ckpt_path, map_location='cpu')
        model.load_state_dicts(ckpt_stage1)
        del ckpt_stage1
        gc.collect()
        print(f"    [OK] Successfully initialized from Stage 1 checkpoint!", flush=True)
    else:
        print(f"[WARNING] Stage 1 checkpoint NOT found at {base_ckpt_path}! Starting without Stage 1 initialization.", flush=True)

    n_params = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"  trainable params: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    last_path = os.path.join(CKPT_DIR, f'checkpoint_{exp}_last.pt')
    best_path = os.path.join(CKPT_DIR, f'best_model_{exp}.pt')
    start_epoch, best_val = 0, float('inf')
    phase_start_epoch = 0
    phase_target_epochs = args.epochs
    loaded_sched_state = None

    # Resume Stage 2 if existing checkpoint exists
    if os.path.exists(last_path) and not smoke:
        print(f"[*] Resuming Stage 2 from {last_path}")
        ckpt = torch.load(last_path, map_location='cpu')
        model.load_state_dicts(ckpt)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        best_val = ckpt.get('best_val_loss', float('inf'))
        prev_target_epochs = ckpt.get('phase_target_epochs', 30)

        if prev_target_epochs != args.epochs and start_epoch < args.epochs:
            # Cosine Warm-Restart for extended Stage 2 training (e.g. Epoch 90 -> 120)
            phase_start_epoch = start_epoch
            phase_target_epochs = args.epochs
            if os.path.exists(best_path):
                backup_best = os.path.join(CKPT_DIR, f'best_model_{exp}_ep{start_epoch:02d}.pt')
                if not os.path.exists(backup_best):
                    import shutil
                    shutil.copy2(best_path, backup_best)
            best_val = float('inf')
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
                pg['initial_lr'] = args.lr
            print(f"    [WARM-RESTART] Extending Stage 2 from epoch {start_epoch} -> {args.epochs} "
                  f"(New cosine cycle: peak lr={args.lr:.1e}, warmup={args.warmup_epochs} ep, best val reset)")
        else:
            phase_start_epoch = ckpt.get('phase_start_epoch', 0)
            phase_target_epochs = prev_target_epochs
            loaded_sched_state = ckpt.get('scheduler_state_dict', None)
            print(f"    resumed at epoch {start_epoch}/{args.epochs} (best val {best_val:.4f})")
        del ckpt
        gc.collect()

    phase_epochs = max(1, phase_target_epochs - phase_start_epoch)
    phase_total_steps = steps_per_epoch * phase_epochs

    def lr_lambda(step):  # step = optimizer steps taken in current phase
        if step < warmup_steps:
            return 0.01 + 0.99 * (step / max(1, warmup_steps))
        prog = (step - warmup_steps) / max(1, phase_total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return (args.min_lr / args.lr) + (1 - args.min_lr / args.lr) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if loaded_sched_state is not None:
        scheduler.load_state_dict(loaded_sched_state)

    log_path = os.path.join(RESULTS_DIR, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_all', 'val_young', 'val_inter', 'val_senes', 'lr', 'epoch_time_s'])

    # Smoke Test mode
    if smoke:
        print(f"\n[*] Running SMOKE TEST ({args.smoke} micro-batches)...")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        it = iter(train_loader)
        for i in range(args.smoke):
            img, pas, ct = next(it)
            img = img.to(DEVICE, memory_format=torch.channels_last)
            pas, ct = pas.to(DEVICE), ct.to(DEVICE)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = model(img, pas, ct) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0 or (i + 1) == args.smoke:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_ema()
        val_res = run_validation(model, val_loader, DEVICE, amp_dtype)
        print(f"    smoke val: {val_res}")
        save_sweep(model, val_ds, RESULTS_DIR, 'smoke', DEVICE)
        print(f"[*] Smoke test successful! Visual sweeps saved to {RESULTS_DIR}")
        return

    # Initial validation before training
    if start_epoch == 0:
        print("[*] Running pre-training Stage 2 baseline validation...")
        v0 = run_validation(model, val_loader, DEVICE, amp_dtype)
        print(f"    baseline val: {v0}")
        save_sweep(model, val_ds, RESULTS_DIR, 'epoch_00_baseline', DEVICE)

    # Main Training Loop
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()
        model.train()
        model.vae.eval()
        ep_loss, micro_count = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Ep {epoch:02d}/{args.epochs:02d}", leave=False)
        for step, (img, pas, ct) in enumerate(pbar):
            img = img.to(DEVICE, memory_format=torch.channels_last)
            pas, ct = pas.to(DEVICE), ct.to(DEVICE)

            if amp_dtype == torch.bfloat16:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = model(img, pas, ct)
                loss_scaled = loss / args.accum
                loss_scaled.backward()
            else:
                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = model(img, pas, ct)
                loss_scaled = loss / args.accum
                scaler.scale(loss_scaled).backward()

            ep_loss += float(loss.detach())
            micro_count += 1

            if (step + 1) % args.accum == 0 or (step + 1) == len(train_loader):
                if amp_dtype == torch.bfloat16:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                    optimizer.step()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                model.update_ema()

            if (step + 1) % (10 * args.accum) == 0:
                cur_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({'loss': f"{ep_loss/micro_count:.4f}", 'lr': f"{cur_lr:.2e}"})

        train_loss = ep_loss / max(1, micro_count)
        ep_time = time.time() - t0

        # Periodic Validation
        val_res = {'all': float('nan'), 'young': float('nan'), 'inter': float('nan'), 'senes': float('nan')}
        if (epoch % args.val_freq == 0) or (epoch == args.epochs):
            val_res = run_validation(model, val_loader, DEVICE, amp_dtype)
            is_best = val_res['all'] < best_val
            if is_best:
                best_val = val_res['all']

            cur_lr = scheduler.get_last_lr()[0]
            print(f"Ep {epoch:02d}/{args.epochs:02d} | train {train_loss:.4f} | "
                  f"val all {val_res['all']:.4f} (y {val_res['young']:.4f}, i {val_res['inter']:.4f}, s {val_res['senes']:.4f}) | "
                  f"lr {cur_lr:.2e} | best {best_val:.4f}{' *BEST*' if is_best else ''} | {ep_time:.1f}s", flush=True)

            if is_best:
                torch.save({
                    'epoch': epoch,
                    'best_val_loss': best_val,
                    **model.state_dicts(),
                }, best_path)

            if (epoch % (args.val_freq * 2) == 0) or (epoch == args.epochs):
                save_sweep(model, val_ds, RESULTS_DIR, f'epoch_{epoch:02d}', DEVICE)
        else:
            cur_lr = scheduler.get_last_lr()[0]
            print(f"Ep {epoch:02d}/{args.epochs:02d} | train {train_loss:.4f} | lr {cur_lr:.2e} | {ep_time:.1f}s", flush=True)

        # Logging
        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, f"{train_loss:.5f}",
                f"{val_res['all']:.5f}", f"{val_res['young']:.5f}",
                f"{val_res['inter']:.5f}", f"{val_res['senes']:.5f}",
                f"{cur_lr:.3e}", f"{ep_time:.1f}"
            ])

        # Save checkpoint last
        state = {
            'epoch': epoch,
            'best_val_loss': best_val,
            'phase_start_epoch': phase_start_epoch,
            'phase_target_epochs': phase_target_epochs,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            **model.state_dicts(),
        }
        torch.save(state, last_path)
        if epoch % args.save_interval == 0:
            torch.save(state, os.path.join(CKPT_DIR, f'checkpoint_{exp}_epoch_{epoch:02d}.pt'))

    print(f"\n{'='*78}\n   LDM v16 STAGE 2 COMPLETED! Best Val Loss: {best_val:.5f}\n   Checkpoints: {CKPT_DIR}\n{'='*78}\n")


if __name__ == '__main__':
    main()
