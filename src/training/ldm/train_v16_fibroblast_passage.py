"""
LDM v16 - Stage 1: Fibroblast Passage-Conditioned Training (from scratch)
=========================================================================
  - Data:        14,820 Picard fibroblast tiles (deneme/processed_tiles), real passage labels P1-P42
  - Split:       donor-level; hFB14 held out for validation (same unseen donor as classifier v3)
  - Conditioning: continuous passage (sinusoidal + MLP) + cell_type embedding (0 = Fibroblast)
  - Sampling:    passage-balanced (inverse-sqrt frequency), aug = flips + rot90, passage jitter +-0.25
  - Optimization: AdamW, warmup + cosine (per optimizer step), bf16 autocast, EMA (warm-up)
  - Seed:        2026
  - Outputs:     checkpoints/ldm/checkpoint_<exp>_last.pt, best_model_<exp>.pt,
                 results/generated/ldm/<exp>/ (passage sweeps), results/generated/ldm/<exp>/train_log.csv

Usage:
    python src/training/ldm/train_v16_fibroblast_passage.py                 # full run (resumes automatically)
    python src/training/ldm/train_v16_fibroblast_passage.py --smoke 30      # throughput / sanity test
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
from models.ldm.dataset_v16 import FibroblastPassageDataset

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('medium')

SWEEP_PASSAGES = [1, 8, 16, 24, 32, 41]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', default='v16_fibroblast_passage')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--warmup_epochs', type=float, default=2.0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--lpips_weight', type=float, default=0.1)
    p.add_argument('--cfg_drop', type=float, default=0.10)
    p.add_argument('--val_donor', default='hFB14')
    p.add_argument('--val_stride', type=int, default=3, help='use every N-th val tile for val loss')
    p.add_argument('--val_freq', type=int, default=2)
    p.add_argument('--save_interval', type=int, default=10)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--smoke', type=int, default=0,
                   help='run only N micro-batches + one val pass + one sweep, then exit (isolated exp name)')
    return p.parse_args()


@torch.no_grad()
def run_validation(model, val_loader, device, amp_dtype):
    """Deterministic val epsilon-MSE (fixed RNG, EMA weights), reported overall and per passage group."""
    model.unet.eval(); model.cond.eval()
    sums = {'all': 0.0, 'young': 0.0, 'inter': 0.0, 'senes': 0.0}
    cnts = {k: 0 for k in sums}
    with torch.random.fork_rng(devices=[0] if device == 'cuda' else []):
        torch.manual_seed(1234)
        for img, pas, ct in val_loader:
            img = img.to(device, memory_format=torch.channels_last)
            pas, ct = pas.to(device), ct.to(device)
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                l = model(img, pas, ct, per_sample=True, use_ema=True)
            l = l.float().cpu().numpy(); p = pas.cpu().numpy()
            for g, m in (('all', np.ones_like(p, bool)), ('young', p <= 8),
                         ('inter', (p > 8) & (p < 20)), ('senes', p >= 20)):
                sums[g] += float(l[m].sum()); cnts[g] += int(m.sum())
    return {k: (sums[k] / cnts[k] if cnts[k] else float('nan')) for k in sums}


@torch.no_grad()
def save_sweep(model, val_ds, results_dir, tag, device):
    """Fixed-noise passage sweep (noise->image) + SDEdit passage translation of a real held-out tile."""
    g = torch.Generator(device=device).manual_seed(0)
    noise = torch.randn(1, 4, 64, 64, device=device, generator=g).repeat(len(SWEEP_PASSAGES), 1, 1, 1)
    imgs = model.sample(SWEEP_PASSAGES, noise=noise, guidance_scale=3.0, num_steps=50)
    save_image(imgs.cpu(), os.path.join(results_dir, f'{tag}_sweep_noise2img.png'), nrow=len(SWEEP_PASSAGES))

    lo = int(np.argmin(val_ds.passages))
    real = val_ds[lo][0].unsqueeze(0).to(device)
    real_vis = (real.clamp(-1, 1) + 1) / 2
    rows = [real_vis.cpu()]
    for tp in SWEEP_PASSAGES[1:]:
        rows.append(model.translate(real, tp, strength=0.7, guidance_scale=3.0, num_steps=50).cpu())
    save_image(torch.cat(rows, 0), os.path.join(results_dir, f'{tag}_sweep_sdedit.png'), nrow=len(rows))


def main():
    args = parse_args()
    smoke = args.smoke > 0
    exp = 'v16_smoke' if smoke else args.exp_name

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    amp_dtype = torch.bfloat16 if (DEVICE == 'cuda' and torch.cuda.is_bf16_supported()) else torch.float16

    CKPT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
    RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', exp)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_ds = FibroblastPassageDataset(root_dir, 'train', val_donor=args.val_donor)
    val_ds = FibroblastPassageDataset(root_dir, 'val', val_donor=args.val_donor)
    val_subset = torch.utils.data.Subset(val_ds, list(range(0, len(val_ds), args.val_stride)))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_ds.make_sampler(args.seed),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(2, args.num_workers), pin_memory=True)

    steps_per_epoch = len(train_loader) // args.accum
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(steps_per_epoch * args.warmup_epochs)

    print(f"\n{'='*78}\n   LDM v16 STAGE 1 | FIBROBLAST PASSAGE-CONDITIONED | exp={exp}\n{'='*78}")
    print(f"  tiles train/val: {len(train_ds)}/{len(val_ds)} (val subset {len(val_subset)}) | donor holdout: {args.val_donor}")
    print(f"  batch {args.batch_size} x accum {args.accum} = eff {args.batch_size*args.accum} | "
          f"{steps_per_epoch} opt-steps/epoch | {total_steps} total | warmup {warmup_steps}")
    print(f"  lr {args.lr} -> {args.min_lr} | wd {args.weight_decay} | lpips_w {args.lpips_weight} | "
          f"cfg_drop {args.cfg_drop} | amp {amp_dtype} | seed {args.seed}\n{'='*78}\n")

    model = CellLDM_v16(cfg_drop_prob=args.cfg_drop, lpips_weight=args.lpips_weight)
    model.to(DEVICE)
    model.unet.to(memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema()
    n_params = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"  trainable params: {n_params:.1f}M")

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(step):  # step = optimizer steps taken
        if step < warmup_steps:
            return 0.01 + 0.99 * (step / max(1, warmup_steps))
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return (args.min_lr / args.lr) + (1 - args.min_lr / args.lr) * cos

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    last_path = os.path.join(CKPT_DIR, f'checkpoint_{exp}_last.pt')
    best_path = os.path.join(CKPT_DIR, f'best_model_{exp}.pt')
    start_epoch, best_val = 0, float('inf')

    if os.path.exists(last_path) and not smoke:
        print(f"[*] Resuming from {last_path}")
        ckpt = torch.load(last_path, map_location='cpu')
        model.load_state_dicts(ckpt)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val = ckpt.get('best_val_loss', float('inf'))
        print(f"    resumed at epoch {start_epoch} (best val {best_val:.4f})")
        del ckpt; gc.collect()

    log_path = os.path.join(RESULTS_DIR, 'train_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'val_all', 'val_young', 'val_inter', 'val_senes', 'lr', 'sec'])

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.unet.train(); model.cond.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, n = 0.0, 0
        n_micro = steps_per_epoch * args.accum
        if smoke:
            n_micro = min(n_micro, args.smoke)

        pbar = tqdm(enumerate(train_loader), total=n_micro, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (img, pas, ct) in pbar:
            if step >= n_micro:
                break
            img = img.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            pas, ct = pas.to(DEVICE), ct.to(DEVICE)

            with torch.amp.autocast('cuda', dtype=amp_dtype):
                loss = model(img, pas, ct)
            (loss / args.accum).backward()
            run_loss += loss.item(); n += 1

            if (step + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                model.update_ema()
            pbar.set_postfix(loss=f"{run_loss/n:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        train_loss = run_loss / max(1, n)
        cur_lr = optimizer.param_groups[0]['lr']
        sec = time.time() - t0
        if smoke:
            print(f"\n[smoke] {n} micro-batches in {sec:.1f}s -> {sec/max(1,n):.2f}s/micro-batch | "
                  f"est. {sec/max(1,n)*steps_per_epoch*args.accum/60:.1f} min/epoch | "
                  f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")

        do_val = smoke or (epoch + 1) % args.val_freq == 0 or (epoch + 1) == args.epochs
        v = {'all': float('nan'), 'young': float('nan'), 'inter': float('nan'), 'senes': float('nan')}
        if do_val:
            v = run_validation(model, val_loader, DEVICE, amp_dtype)
            print(f"[Epoch {epoch+1}/{args.epochs}] train {train_loss:.4f} | val {v['all']:.4f} "
                  f"(young {v['young']:.4f} / inter {v['inter']:.4f} / senes {v['senes']:.4f}) | lr {cur_lr:.2e} | {sec/60:.1f} min")
            try:
                save_sweep(model, val_ds, RESULTS_DIR, f'epoch_{epoch+1}', DEVICE)
                print(f"  [+] passage sweeps saved -> {RESULTS_DIR}")
            except Exception as e:
                print(f"  [!] sweep generation failed: {e}")
            if not smoke and v['all'] < best_val:
                best_val = v['all']
                torch.save({'epoch': epoch + 1, 'best_val_loss': best_val, **model.state_dicts()}, best_path)
                print(f"  [*] best model saved (val {best_val:.4f})")
        else:
            print(f"[Epoch {epoch+1}/{args.epochs}] train {train_loss:.4f} | lr {cur_lr:.2e} | {sec/60:.1f} min")

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f"{train_loss:.5f}", f"{v['all']:.5f}", f"{v['young']:.5f}",
                                    f"{v['inter']:.5f}", f"{v['senes']:.5f}", f"{cur_lr:.3e}", f"{sec:.0f}"])
        if smoke:
            print("[smoke] OK - exiting.")
            return

        if (epoch + 1) % args.save_interval == 0:
            torch.save({'epoch': epoch + 1, **model.state_dicts()},
                       os.path.join(CKPT_DIR, f'checkpoint_{exp}_epoch_{epoch+1}.pt'))
        torch.save({'epoch': epoch + 1, 'best_val_loss': best_val, 'args': vars(args),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), **model.state_dicts()}, last_path)

    print(f"\n{'='*78}\n[DONE] LDM v16 Stage 1 finished. Weights: {last_path}\n{'='*78}\n")


if __name__ == '__main__':
    main()
