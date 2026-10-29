"""
Continuous 5-Step Passage Trajectory Evaluation (LDM v16 Stage 2 MSC)
======================================================================
Evaluates gradual passage translation (-1 or +1 passage per step across 5 steps)
using LDM v16 Stage 2 canonical Low-FID parameters with fixed latent noise.

Modes:
  - Rejuvenation: Senescent test crops (N=318) -> P_in - 1..5 (clamped to P=3.5)
  - Aging:         Young test crops (N=327)     -> P_in + 1..5 (clamped to P=15.5)
  - Both:          Runs rejuvenation and aging consecutively.

Metrics computed per step:
  1. ConvNeXt-Tiny classifier v3: Mean / Median predicted passage & Delta P.
  2. ResNet-18 classifier (native 512x512): Target accuracy (%) & confidence (%).
  3. Official FID score vs real reference cohort (pytorch-fid).
  4. Mean brightness monitoring (verifying absence of contrast/whitening shift).

Outputs (saved to results/v16_stage2_evaluation/<direction>_trajectory/):
  - gen_step1..5/*.png
  - <direction>_trajectory_montage.png
  - <direction>_trajectory_curves.png
  - <direction>_trajectory_summary.csv
  - <direction>_trajectory_per_sample.csv
"""

import argparse
import os
import sys
import time
import gc

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from pytorch_fid import fid_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v16 import CellLDM_v16, CELL_MSC
from models.classifier.classifier import Classifier

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


# =========================================================================
# Classifiers
# =========================================================================
class ConvNeXtMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[2].in_features
        self.norm = base.classifier[0]
        self.age_head = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.phase_head = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, 128), nn.LayerNorm(128),
            nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.norm(self.avgpool(self.features(x)))
        return self.age_head(x).squeeze(1), self.phase_head(x)


def load_convnext_classifier():
    cls = ConvNeXtMultiTask().to(dev)
    ckpt_path = os.path.join(root_dir, 'deneme', 'checkpoints', 'classifier_v3_convnext_best.pth')
    ck = torch.load(ckpt_path, map_location=dev)
    cls.load_state_dict(ck['model_state_dict'])
    cls.eval()
    return cls


def load_resnet_classifier():
    cls_r = Classifier(output_size=2).to(dev)
    ckpt_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    ck_r = torch.load(ckpt_path, map_location=dev)
    cls_r.load_state_dict(ck_r)
    cls_r.eval()
    return cls_r


@torch.no_grad()
def predict_passages(cls_convnext, imgs_01, bs=32):
    """Predicts continuous passage values using ConvNeXt-Tiny v3 resized to 224x224."""
    preds = []
    for i in range(0, len(imgs_01), bs):
        batch = imgs_01[i:i + bs].to(dev)
        g = transforms.functional.resize(batch, (224, 224), antialias=True)
        g = (g - IMNET_MEAN) / IMNET_STD
        ages, _ = cls_convnext(g)
        p = ages.cpu().numpy() * 40.0 + 1.0
        preds.extend(p.tolist())
    return np.array(preds)


@torch.no_grad()
def predict_resnet18(cls_resnet, imgs_01, bs=32):
    """Evaluates binary ResNet-18 at native 512x512 resolution. Class 0: Senescent, Class 1: Young."""
    preds_bin = []
    probs_young = []
    for i in range(0, len(imgs_01), bs):
        batch = imgs_01[i:i + bs].to(dev)
        g = (batch - IMNET_MEAN) / IMNET_STD
        logits = cls_resnet(g)
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).cpu().numpy()
        young_p = probs[:, 1].cpu().numpy()
        preds_bin.extend(pred_class.tolist())
        probs_young.extend(young_p.tolist())
    return np.array(preds_bin), np.array(probs_young)


# =========================================================================
# Trajectory Execution
# =========================================================================
def run_trajectory(direction, args):
    is_rejuv = (direction.lower() == 'rejuvenation')
    dir_name = 'rejuvenation_trajectory' if is_rejuv else 'aging_trajectory'
    if getattr(args, 'out_dir', None):
        out_base = os.path.join(args.out_dir, dir_name)
    else:
        out_base = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', dir_name)
    os.makedirs(out_base, exist_ok=True)

    strength = args.rejuv_strength if is_rejuv else args.aging_strength
    cfg = args.rejuv_cfg if is_rejuv else args.aging_cfg
    step_sign = -1.0 if is_rejuv else +1.0
    clamp_val = args.min_passage if is_rejuv else args.max_passage

    y_test_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    s_test_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')

    in_dir = s_test_dir if is_rejuv else y_test_dir
    ref_dir = y_test_dir if is_rejuv else s_test_dir

    in_files = sorted([os.path.join(in_dir, f) for f in os.listdir(in_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not args.all and args.n_samples is not None:
        in_files = in_files[:args.n_samples]

    n_total = len(in_files)
    all_filenames = [os.path.basename(f) for f in in_files]

    step_dirs = {}
    for s in range(1, args.n_steps + 1):
        sd = os.path.join(out_base, f"gen_step{s}")
        os.makedirs(sd, exist_ok=True)
        step_dirs[s] = sd

    print(f"\n{'='*75}")
    print(f"   STARTING 5-STEP {direction.upper()} TRAJECTORY (N = {n_total} Cells)")
    print(f"   Strength: {strength} | CFG: {cfg} | Step Delta: {step_sign * args.delta_step:+.1f} P | Clamp: {clamp_val}")
    print(f"   Output: {out_base}")
    print(f"{'='*75}\n", flush=True)

    # 1. Baseline input passages
    print("[*] Computing baseline passages for input cells...", flush=True)
    cls_convnext = load_convnext_classifier()
    real_input_tensors = [TF.to_tensor(Image.open(f).convert('RGB')) for f in in_files]
    real_p_list = []
    for i in range(0, n_total, 32):
        chunk = torch.stack(real_input_tensors[i:i + 32])
        real_p_list.extend(predict_passages(cls_convnext, chunk).tolist())
    real_passages = np.array(real_p_list)
    print(f"[OK] Baseline passage: Mean P{real_passages.mean():.2f} +/- {real_passages.std():.2f} (Median: P{np.median(real_passages):.2f})", flush=True)

    del cls_convnext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. Diffusion generation (or skip if existing)
    if args.skip_generation:
        print("[*] --skip_generation specified. Using existing images in gen_step1..5.", flush=True)
        montage_exemplars = []
        for fn in all_filenames[:10]:
            row = [TF.to_tensor(Image.open(os.path.join(in_dir, fn)).convert('RGB'))]
            for s in range(1, args.n_steps + 1):
                row.append(TF.to_tensor(Image.open(os.path.join(step_dirs[s], fn)).convert('RGB')))
            montage_exemplars.append(row)
    else:
        ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(root_dir, 'checkpoints', 'ldm', args.ckpt)
        print(f"[*] Loading LDM v16 Checkpoint: {ckpt_path}", flush=True)
        model = CellLDM_v16().to(dev)
        model.load_state_dicts(torch.load(ckpt_path, map_location='cpu'))
        model.eval()

        bs = args.batch_size
        n_batches = (n_total + bs - 1) // bs
        montage_exemplars = []
        t_start = time.time()

        for b_idx in range(n_batches):
            b_slice = slice(b_idx * bs, min((b_idx + 1) * bs, n_total))
            b_paths = in_files[b_slice]
            b_p_in = real_passages[b_slice]
            b_tensors = torch.stack(real_input_tensors[b_slice])
            current_bs = len(b_paths)

            batch_already_done = all(
                os.path.exists(os.path.join(step_dirs[s], os.path.basename(p)))
                for s in range(1, args.n_steps + 1)
                for p in b_paths
            )
            if batch_already_done:
                if len(montage_exemplars) < 10:
                    for k in range(min(current_bs, 10 - len(montage_exemplars))):
                        fname = os.path.basename(b_paths[k])
                        cell_row = [b_tensors[k]]
                        for s in range(1, args.n_steps + 1):
                            step_img = TF.to_tensor(Image.open(os.path.join(step_dirs[s], fname)).convert('RGB'))
                            cell_row.append(step_img)
                        montage_exemplars.append(cell_row)
                continue

            bx = (b_tensors * 2.0 - 1.0).to(dev)
            with torch.no_grad():
                latents = model.encode(bx)
                torch.manual_seed(args.seed + b_idx)
                fixed_noise = torch.randn_like(latents)
                batch_row_imgs = [b_tensors]

                for s in range(1, args.n_steps + 1):
                    if is_rejuv:
                        target_p = np.maximum(clamp_val, b_p_in - (s * args.delta_step))
                    else:
                        target_p = np.minimum(clamp_val, b_p_in + (s * args.delta_step))
                    target_p_tensor = torch.tensor(target_p, dtype=torch.float32, device=dev)

                    out = model.translate(bx, target_passage=target_p_tensor, cell_type=CELL_MSC,
                                          strength=strength, guidance_scale=cfg,
                                          num_steps=50, use_ema=True, noise=fixed_noise)
                    out_cpu = out.cpu()

                    for k, img_t in enumerate(out_cpu):
                        fname = os.path.basename(b_paths[k])
                        save_image(img_t, os.path.join(step_dirs[s], fname))

                    batch_row_imgs.append(out_cpu)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if len(montage_exemplars) < 10:
                for k in range(min(current_bs, 10 - len(montage_exemplars))):
                    cell_row = [batch_row_imgs[step_idx][k] for step_idx in range(args.n_steps + 1)]
                    montage_exemplars.append(cell_row)

            elapsed = time.time() - t_start
            speed = (b_idx + 1) / elapsed
            remain = (n_batches - b_idx - 1) / max(speed, 1e-5)
            print(f"  [Batch {b_idx + 1:2d}/{n_batches}] Processed {current_bs} cells through 5 steps | Elapsed: {elapsed:.0f}s, ETA: {remain:.0f}s", flush=True)

        print(f"\n[*] Generation complete across {n_total} cells in {time.time() - t_start:.1f}s!", flush=True)
        del model
        if 'latents' in locals():
            del latents, bx
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 3. Classifier evaluation
    print("\n[*] Evaluating classifiers across all steps...", flush=True)
    cls_convnext = load_convnext_classifier()
    cls_resnet = load_resnet_classifier()

    step_preds_convnext = {0: real_passages}
    step_preds_resnet_bin = {}
    step_preds_resnet_target_conf = {}
    step_brightness = {}

    all_real_stacked = torch.stack(real_input_tensors)
    r_bin_0, r_p_0 = predict_resnet18(cls_resnet, all_real_stacked)
    step_preds_resnet_bin[0] = r_bin_0
    step_preds_resnet_target_conf[0] = r_p_0 if is_rejuv else (1.0 - r_p_0)
    step_brightness[0] = all_real_stacked.mean(dim=(1, 2, 3)).numpy()

    for s in range(1, args.n_steps + 1):
        step_files = [os.path.join(step_dirs[s], fn) for fn in all_filenames]
        step_tensors = torch.stack([TF.to_tensor(Image.open(f).convert('RGB')) for f in step_files])
        step_preds_convnext[s] = predict_passages(cls_convnext, step_tensors)
        r_bin_s, r_p_s = predict_resnet18(cls_resnet, step_tensors)
        step_preds_resnet_bin[s] = r_bin_s
        step_preds_resnet_target_conf[s] = r_p_s if is_rejuv else (1.0 - r_p_s)
        step_brightness[s] = step_tensors.mean(dim=(1, 2, 3)).numpy()

        target_acc = np.mean(r_bin_s == (1 if is_rejuv else 0)) * 100.0
        print(f"  [Step {s}] ConvNeXt Mean Passage: P{step_preds_convnext[s].mean():.2f} | ResNet-18 Target Acc: {target_acc:.1f}%", flush=True)

    del cls_convnext, cls_resnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Official FID calculation
    print("\n[*] Calculating official FID scores vs reference test cohort...", flush=True)
    step_fids = {}
    try:
        step_fids[0] = float(fid_score.calculate_fid_given_paths([ref_dir, in_dir], batch_size=32, device=dev, dims=2048))
        print(f"  -> Step 0 Baseline FID: {step_fids[0]:.2f}", flush=True)
    except Exception as e:
        print(f"  [!] FID Step 0 error: {e}")
        step_fids[0] = -1.0

    for s in range(1, args.n_steps + 1):
        try:
            fid_val = float(fid_score.calculate_fid_given_paths([ref_dir, step_dirs[s]], batch_size=32, device=dev, dims=2048))
            step_fids[s] = fid_val
            print(f"  -> Step {s} Low-FID: {fid_val:.2f}", flush=True)
        except Exception as e:
            print(f"  [!] FID Step {s} error: {e}")
            step_fids[s] = -1.0

    # 5. Summary CSV
    real_p = step_preds_convnext[0]
    target_cls_idx = 1 if is_rejuv else 0
    acc_col_name = 'ResNet18_Young_Acc_pct' if is_rejuv else 'ResNet18_Senescent_Acc_pct'
    conf_col_name = 'ResNet18_Young_Conf_pct' if is_rejuv else 'ResNet18_Senescent_Conf_pct'
    success_col_name = 'Rejuvenation_Success_pct' if is_rejuv else 'Aging_Success_pct'

    summary_rows = []
    for s in range(args.n_steps + 1):
        p_arr = step_preds_convnext[s]
        r_bin = step_preds_resnet_bin[s]
        r_conf = step_preds_resnet_target_conf[s]
        b_arr = step_brightness[s]

        delta_p = float(np.mean(p_arr - real_p)) if s > 0 else 0.0
        if is_rejuv:
            success_pct = float(np.mean(p_arr < real_p) * 100.0) if s > 0 else 0.0
        else:
            success_pct = float(np.mean(p_arr > real_p) * 100.0) if s > 0 else 0.0

        target_acc_pct = float(np.mean(r_bin == target_cls_idx) * 100.0)
        target_conf_pct = float(np.mean(r_conf) * 100.0)
        bright_shift = float(np.mean(b_arr - step_brightness[0])) if s > 0 else 0.0

        label = f"Step 0 (Real {'Senescent' if is_rejuv else 'Young'})" if s == 0 else f"Step {s} (Target P {step_sign * s * args.delta_step:+.1f})"

        summary_rows.append({
            'Step': s,
            'Label': label,
            'FID': round(step_fids[s], 2) if step_fids[s] > 0 else 'N/A',
            'ConvNeXt_Passage_Mean': round(float(np.mean(p_arr)), 2),
            'ConvNeXt_Passage_Median': round(float(np.median(p_arr)), 2),
            'Delta_Passage': round(delta_p, 2),
            success_col_name: round(success_pct, 1),
            acc_col_name: round(target_acc_pct, 1),
            conf_col_name: round(target_conf_pct, 1),
            'Mean_Brightness': round(float(np.mean(b_arr)), 4),
            'Brightness_Shift': round(bright_shift, 4),
        })

    df_summary = pd.DataFrame(summary_rows)
    prefix = 'rejuvenation' if is_rejuv else 'aging'
    summary_csv_path = os.path.join(out_base, f'{prefix}_trajectory_summary.csv')
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"\n[OK] Summary CSV saved: {summary_csv_path}", flush=True)

    # Per-sample CSV
    conf_key = 'young_conf' if is_rejuv else 'senescent_conf'
    sample_dict = {'filename': all_filenames, 'real_passage': np.round(real_p, 2)}
    for s in range(1, args.n_steps + 1):
        sample_dict[f'step{s}_passage'] = np.round(step_preds_convnext[s], 2)
        sample_dict[f'step{s}_{conf_key}'] = np.round(step_preds_resnet_target_conf[s], 3)
    df_samples = pd.DataFrame(sample_dict)
    samples_csv_path = os.path.join(out_base, f'{prefix}_trajectory_per_sample.csv')
    df_samples.to_csv(samples_csv_path, index=False)
    print(f"[OK] Per-sample CSV saved: {samples_csv_path}", flush=True)

    # 6. Continuum Montage
    print("[*] Generating side-by-side continuum montage...", flush=True)
    montage_rows = [torch.cat(row, dim=2) for row in montage_exemplars]
    montage_tensor = torch.cat(montage_rows, dim=1)
    montage_img_path = os.path.join(out_base, f'{prefix}_trajectory_montage.png')
    save_image(montage_tensor, montage_img_path)
    print(f"[OK] Continuum montage saved: {montage_img_path}", flush=True)

    # 7. Matplotlib Publication Curves
    print("[*] Generating trajectory curves figure...", flush=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    steps = list(range(args.n_steps + 1))
    step_labels = ['Real\n(P_in)'] + [f'S{s}\n(P{step_sign * s * args.delta_step:+.0f})' for s in range(1, args.n_steps + 1)]

    # 1. Passage
    ax = axes[0]
    p_means = df_summary['ConvNeXt_Passage_Mean'].values
    p_medians = df_summary['ConvNeXt_Passage_Median'].values
    c_line = '#1f77b4' if is_rejuv else '#d62728'
    c_fill = '#aec7e8' if is_rejuv else '#ff9896'
    ax.plot(steps, p_means, 'o-', color=c_line, linewidth=2.5, markersize=8, label='Mean Passage')
    ax.plot(steps, p_medians, 's--', color=c_fill, linewidth=2, markersize=7, label='Median Passage')
    ax.set_xticks(steps)
    ax.set_xticklabels(step_labels, fontsize=10)
    ax.set_ylabel('ConvNeXt-Tiny v3 Passage', fontsize=12, fontweight='bold')
    ax.set_title(f"Passage {'Reduction' if is_rejuv else 'Progression'} Trajectory", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    # 2. ResNet-18 Target Accuracy
    ax = axes[1]
    r_accs = df_summary[acc_col_name].values
    r_confs = df_summary[conf_col_name].values
    c_acc = '#2ca02c' if is_rejuv else '#ff7f0e'
    c_conf = '#98df8a' if is_rejuv else '#ffbb78'
    ax.plot(steps, r_accs, 'o-', color=c_acc, linewidth=2.5, markersize=8, label='Target Accuracy (%)')
    ax.plot(steps, r_confs, '^--', color=c_conf, linewidth=2, markersize=7, label='Target Confidence (%)')
    ax.set_xticks(steps)
    ax.set_xticklabels(step_labels, fontsize=10)
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f"ResNet-18 {'Young' if is_rejuv else 'Senescent'} Classification", fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    # 3. FID
    ax = axes[2]
    fids = [f if isinstance(f, (int, float)) and f > 0 else np.nan for f in df_summary['FID'].values]
    c_fid = '#d62728' if is_rejuv else '#9467bd'
    ax.plot(steps, fids, 'o-', color=c_fid, linewidth=2.5, markersize=8, label=f"FID vs {'Young' if is_rejuv else 'Senescent'} Reference")
    ax.set_xticks(steps)
    ax.set_xticklabels(step_labels, fontsize=10)
    ax.set_ylabel('FID (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Distributional Convergence (FID)', fontsize=13, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    curves_img_path = os.path.join(out_base, f'{prefix}_trajectory_curves.png')
    plt.savefig(curves_img_path)
    plt.close()
    print(f"[OK] Trajectory curves saved: {curves_img_path}\n", flush=True)


def main():
    p = argparse.ArgumentParser(description='LDM v16 5-Step Continuous Trajectory Evaluation')
    p.add_argument('--direction', choices=['rejuvenation', 'aging', 'both'], default='both',
                   help='Trajectory direction to evaluate')
    p.add_argument('--ckpt', default='best_model_v16_stage2_msc.pt',
                   help='Checkpoint filename in checkpoints/ldm or absolute path')
    p.add_argument('--out_dir', default=None,
                   help='Custom root directory for outputs (defaults to results/v16_stage2_evaluation)')
    p.add_argument('--all', action='store_true', default=False,
                   help='Process full test cohort (Senescent N=318, Young N=327)')
    p.add_argument('--n_samples', type=int, default=None,
                   help='Subset number of crops if specified')
    p.add_argument('--batch_size', type=int, default=8, help='Diffusion batch size')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--rejuv_strength', type=float, default=0.65)
    p.add_argument('--rejuv_cfg', type=float, default=3.5)
    p.add_argument('--aging_strength', type=float, default=0.75)
    p.add_argument('--aging_cfg', type=float, default=4.0)
    p.add_argument('--n_steps', type=int, default=5, help='Number of gradual trajectory steps')
    p.add_argument('--delta_step', type=float, default=1.0, help='Passage delta per step')
    p.add_argument('--min_passage', type=float, default=3.5, help='Biological young median clamp')
    p.add_argument('--max_passage', type=float, default=15.5, help='Biological senescent mean clamp')
    p.add_argument('--skip_generation', action='store_true',
                   help='Skip generation if images already exist and re-run evaluation only')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    directions = ['rejuvenation', 'aging'] if args.direction == 'both' else [args.direction]
    for d in directions:
        run_trajectory(d, args)


if __name__ == '__main__':
    main()
