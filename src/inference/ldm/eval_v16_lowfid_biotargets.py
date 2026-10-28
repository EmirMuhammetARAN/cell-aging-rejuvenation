"""
Evaluation of LDM v16 Stage 2 with Canonical Low-FID Parameters & Biological Target Passages
=============================================================================================
Uses strictly the benchmark Low-FID parameters on full canvas (unmasked):
  - Rejuvenation: Real Senescent test cells -> Target Passage P = 3.5 (Young MSC Median)
                  Parameters: Strength = 0.65, CFG = 3.5
  - Aging:         Real Young test cells -> Target Passage P = 15.5 (Senescent MSC Median)
                  Parameters: Strength = 0.75, CFG = 4.0

Outputs:
  - results/v16_stage2_evaluation/lowfid_biotargets/gen_rejuv/*.png
  - results/v16_stage2_evaluation/lowfid_biotargets/gen_aging/*.png
  - results/v16_stage2_evaluation/lowfid_biotargets/rejuvenation_lowfid_montage.png
  - results/v16_stage2_evaluation/lowfid_biotargets/aging_lowfid_montage.png
  - results/v16_stage2_evaluation/lowfid_biotargets/rejuvenation_predictions.csv
  - results/v16_stage2_evaluation/lowfid_biotargets/aging_predictions.csv
  - results/v16_stage2_evaluation/lowfid_biotargets/v16_stage2_lowfid_summary.csv
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
from PIL import Image
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from pytorch_fid import fid_score

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v16 import CellLDM_v16, CELL_MSC

dev = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNeXtMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[2].in_features
        self.norm = base.classifier[0]
        self.age_head = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 128), nn.LayerNorm(128),
                                      nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
        self.phase_head = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 128), nn.LayerNorm(128),
                                        nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))

    def forward(self, x):
        x = self.norm(self.avgpool(self.features(x)))
        return self.age_head(x).squeeze(1), self.phase_head(x)


def load_classifier():
    cls = ConvNeXtMultiTask().to(dev)
    ck = torch.load(os.path.join(root_dir, 'deneme', 'checkpoints', 'classifier_v3_convnext_best.pth'), map_location=dev)
    cls.load_state_dict(ck['model_state_dict'])
    cls.eval()
    return cls


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(dev)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(dev)


@torch.no_grad()
def predict_passages(cls, imgs_01, bs=32):
    preds = []
    for i in range(0, len(imgs_01), bs):
        batch = imgs_01[i:i + bs].to(dev)
        g = transforms.functional.resize(batch, (224, 224), antialias=True)
        g = (g - IMNET_MEAN) / IMNET_STD
        ages, _ = cls(g)
        p = ages.cpu().numpy() * 40.0 + 1.0
        preds.extend(p.tolist())
    return np.array(preds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n_samples', type=int, default=32, help='Number of test cells per direction')
    p.add_argument('--all', action='store_true', help='Process all test crops')
    p.add_argument('--batch_size', type=int, default=8, help='Generation batch size')
    p.add_argument('--seed', type=int, default=2026)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', 'lowfid_biotargets')
    gen_rejuv_dir = os.path.join(out_dir, 'gen_rejuv')
    gen_aging_dir = os.path.join(out_dir, 'gen_aging')
    os.makedirs(gen_rejuv_dir, exist_ok=True)
    os.makedirs(gen_aging_dir, exist_ok=True)

    print("[*] Loading LDM v16 Stage 2 Model...", flush=True)
    model = CellLDM_v16().to(dev)
    ckpt_path = os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v16_stage2_msc.pt')
    model.load_state_dicts(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    cls = load_classifier()

    # Dataset paths
    y_test_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    s_test_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
    y_files = sorted([os.path.join(y_test_dir, f) for f in os.listdir(y_test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    s_files = sorted([os.path.join(s_test_dir, f) for f in os.listdir(s_test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not args.all:
        y_files = y_files[:args.n_samples]
        s_files = s_files[:args.n_samples]

    print(f"[*] Starting Low-FID Evaluation for {len(s_files)} senescent and {len(y_files)} young test crops...", flush=True)

    bs = args.batch_size

    # =========================================================================
    # 1. REJUVENATION: Senescent -> Target P = 3.5 (Young MSC Median)
    # Low-FID settings: strength = 0.65, CFG = 3.5
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print(f"   1. REJUVENATION: Senescent -> Target P = 3.5 (Strength = 0.65, CFG = 3.5, N={len(s_files)})", flush=True)
    print("=" * 70, flush=True)

    rejuv_gen_tensors = []
    real_s_preds = []
    real_s_bright = []
    p_rejuv_list = []
    b_rejuv_list = []
    s_filenames = [os.path.basename(f) for f in s_files]

    t0 = time.time()
    n_batches_rejuv = (len(s_files) + bs - 1) // bs
    for b_idx in range(n_batches_rejuv):
        batch_paths = s_files[b_idx * bs:(b_idx + 1) * bs]
        batch_tensors = torch.stack([TF.to_tensor(Image.open(f).convert('RGB')) for f in batch_paths])
        
        # Passage and brightness of real input
        preds_in = predict_passages(cls, batch_tensors)
        bright_in = batch_tensors.mean(dim=(1, 2, 3)).numpy()
        real_s_preds.extend(preds_in.tolist())
        real_s_bright.extend(bright_in.tolist())

        # Diffusion translation
        bx = (batch_tensors * 2.0 - 1.0).to(dev)
        with torch.no_grad():
            out = model.translate(bx, target_passage=3.5, cell_type=CELL_MSC,
                                  strength=0.65, guidance_scale=3.5, num_steps=50, use_ema=True)
            out_cpu = out.cpu()

        # Passage and brightness of generated rejuvenation
        preds_out = predict_passages(cls, out_cpu)
        bright_out = out_cpu.mean(dim=(1, 2, 3)).numpy()
        p_rejuv_list.extend(preds_out.tolist())
        b_rejuv_list.extend(bright_out.tolist())

        # Save individual generated images
        for k, img_t in enumerate(out_cpu):
            fname = os.path.basename(batch_paths[k])
            save_image(img_t, os.path.join(gen_rejuv_dir, fname))

        if b_idx < 10:
            for k in range(min(len(batch_paths), 10 - len(rejuv_gen_tensors))):
                rejuv_gen_tensors.append((batch_tensors[k], out_cpu[k]))

        elapsed = time.time() - t0
        speed = (b_idx + 1) / elapsed
        remain = (n_batches_rejuv - b_idx - 1) / max(speed, 1e-5)
        print(f"  [Rejuvenation Batch {b_idx + 1}/{n_batches_rejuv}] "
              f"In: P{preds_in.mean():.1f} -> Out: P{preds_out.mean():.1f} | "
              f"Elapsed: {elapsed:.0f}s, ETA: {remain:.0f}s", flush=True)

    real_s_preds = np.array(real_s_preds)
    real_s_bright = np.array(real_s_bright)
    p_rejuv = np.array(p_rejuv_list)
    b_rejuv = np.array(b_rejuv_list)

    rejuv_success_pct = float(np.mean(p_rejuv < real_s_preds) * 100.0)
    delta_p_rejuv = float(p_rejuv.mean() - real_s_preds.mean())
    bright_shift_rejuv = float(b_rejuv.mean() - real_s_bright.mean())

    print(f"\n[OK] Rejuvenation Completed ({len(s_files)} cells in {time.time() - t0:.1f}s):", flush=True)
    print(f"  Real Senescent Input: Ortalama P{real_s_preds.mean():.2f} +/- {real_s_preds.std():.2f} (Medyan: P{np.median(real_s_preds):.2f})", flush=True)
    print(f"  Low-FID Rejuvenation: Ortalama P{p_rejuv.mean():.2f} +/- {p_rejuv.std():.2f} (Medyan: P{np.median(p_rejuv):.2f})", flush=True)
    print(f"  Pasaj Kaymasi:        {delta_p_rejuv:+.2f} Pasaj", flush=True)
    print(f"  Genclesme Orani:      %{rejuv_success_pct:.1f}", flush=True)
    print(f"  Parlaklik Farkı:      {bright_shift_rejuv:+.4f}", flush=True)

    # Save per-sample Rejuvenation CSV
    df_rejuv_samples = pd.DataFrame({
        'filename': s_filenames,
        'input_passage': np.round(real_s_preds, 2),
        'output_passage': np.round(p_rejuv, 2),
        'delta_passage': np.round(p_rejuv - real_s_preds, 2),
        'rejuvenated': (p_rejuv < real_s_preds).astype(int),
        'input_brightness': np.round(real_s_bright, 4),
        'output_brightness': np.round(b_rejuv, 4),
        'delta_brightness': np.round(b_rejuv - real_s_bright, 4)
    })
    rejuv_csv_path = os.path.join(out_dir, 'rejuvenation_predictions.csv')
    df_rejuv_samples.to_csv(rejuv_csv_path, index=False)
    print(f"  Per-sample CSV saved: {rejuv_csv_path}", flush=True)

    # Save 10-exemplar 2-column montage (Real | Low-FID Rejuvenation)
    rejuv_rows = [torch.cat([pair[0], pair[1]], dim=2) for pair in rejuv_gen_tensors[:10]]
    save_image(torch.cat(rejuv_rows, dim=1), os.path.join(out_dir, 'rejuvenation_lowfid_montage.png'))
    print(f"  Montage saved: {os.path.join(out_dir, 'rejuvenation_lowfid_montage.png')}", flush=True)

    # =========================================================================
    # 2. AGING: Young -> Target P = 15.5 (Senescent MSC Median)
    # Low-FID settings: strength = 0.75, CFG = 4.0
    # =========================================================================
    print("\n" + "=" * 70, flush=True)
    print(f"   2. AGING: Young -> Target P = 15.5 (Strength = 0.75, CFG = 4.0, N={len(y_files)})", flush=True)
    print("=" * 70, flush=True)

    aging_gen_tensors = []
    real_y_preds = []
    real_y_bright = []
    p_aging_list = []
    b_aging_list = []
    y_filenames = [os.path.basename(f) for f in y_files]

    t0_aging = time.time()
    n_batches_aging = (len(y_files) + bs - 1) // bs
    for b_idx in range(n_batches_aging):
        batch_paths = y_files[b_idx * bs:(b_idx + 1) * bs]
        batch_tensors = torch.stack([TF.to_tensor(Image.open(f).convert('RGB')) for f in batch_paths])
        
        # Passage and brightness of real input
        preds_in = predict_passages(cls, batch_tensors)
        bright_in = batch_tensors.mean(dim=(1, 2, 3)).numpy()
        real_y_preds.extend(preds_in.tolist())
        real_y_bright.extend(bright_in.tolist())

        # Diffusion translation
        bx = (batch_tensors * 2.0 - 1.0).to(dev)
        with torch.no_grad():
            out = model.translate(bx, target_passage=15.5, cell_type=CELL_MSC,
                                  strength=0.75, guidance_scale=4.0, num_steps=50, use_ema=True)
            out_cpu = out.cpu()

        # Passage and brightness of generated aging
        preds_out = predict_passages(cls, out_cpu)
        bright_out = out_cpu.mean(dim=(1, 2, 3)).numpy()
        p_aging_list.extend(preds_out.tolist())
        b_aging_list.extend(bright_out.tolist())

        # Save individual generated images
        for k, img_t in enumerate(out_cpu):
            fname = os.path.basename(batch_paths[k])
            save_image(img_t, os.path.join(gen_aging_dir, fname))

        if b_idx < 10:
            for k in range(min(len(batch_paths), 10 - len(aging_gen_tensors))):
                aging_gen_tensors.append((batch_tensors[k], out_cpu[k]))

        elapsed = time.time() - t0_aging
        speed = (b_idx + 1) / elapsed
        remain = (n_batches_aging - b_idx - 1) / max(speed, 1e-5)
        print(f"  [Aging Batch {b_idx + 1}/{n_batches_aging}] "
              f"In: P{preds_in.mean():.1f} -> Out: P{preds_out.mean():.1f} | "
              f"Elapsed: {elapsed:.0f}s, ETA: {remain:.0f}s", flush=True)

    real_y_preds = np.array(real_y_preds)
    real_y_bright = np.array(real_y_bright)
    p_aging = np.array(p_aging_list)
    b_aging = np.array(b_aging_list)

    aging_success_pct = float(np.mean(p_aging > real_y_preds) * 100.0)
    delta_p_aging = float(p_aging.mean() - real_y_preds.mean())
    bright_shift_aging = float(b_aging.mean() - real_y_bright.mean())

    print(f"\n[OK] Aging Completed ({len(y_files)} cells in {time.time() - t0_aging:.1f}s):", flush=True)
    print(f"  Real Young Input:    Ortalama P{real_y_preds.mean():.2f} +/- {real_y_preds.std():.2f} (Medyan: P{np.median(real_y_preds):.2f})", flush=True)
    print(f"  Low-FID Aging:       Ortalama P{p_aging.mean():.2f} +/- {p_aging.std():.2f} (Medyan: P{np.median(p_aging):.2f})", flush=True)
    print(f"  Pasaj Kaymasi:       {delta_p_aging:+.2f} Pasaj", flush=True)
    print(f"  Yaslanma Orani:      %{aging_success_pct:.1f}", flush=True)
    print(f"  Parlaklik Farkı:     {bright_shift_aging:+.4f}", flush=True)

    # Save per-sample Aging CSV
    df_aging_samples = pd.DataFrame({
        'filename': y_filenames,
        'input_passage': np.round(real_y_preds, 2),
        'output_passage': np.round(p_aging, 2),
        'delta_passage': np.round(p_aging - real_y_preds, 2),
        'aged': (p_aging > real_y_preds).astype(int),
        'input_brightness': np.round(real_y_bright, 4),
        'output_brightness': np.round(b_aging, 4),
        'delta_brightness': np.round(b_aging - real_y_bright, 4)
    })
    aging_csv_path = os.path.join(out_dir, 'aging_predictions.csv')
    df_aging_samples.to_csv(aging_csv_path, index=False)
    print(f"  Per-sample CSV saved: {aging_csv_path}", flush=True)

    # Save 10-exemplar 2-column montage (Real | Low-FID Aging)
    aging_rows = [torch.cat([pair[0], pair[1]], dim=2) for pair in aging_gen_tensors[:10]]
    save_image(torch.cat(aging_rows, dim=1), os.path.join(out_dir, 'aging_lowfid_montage.png'))
    print(f"  Montage saved: {os.path.join(out_dir, 'aging_lowfid_montage.png')}", flush=True)

    # =========================================================================
    # 3. FID SCORE CALCULATION
    # =========================================================================
    rejuv_fid = None
    aging_fid = None
    if args.all or len(s_files) >= 30:
        print("\n" + "=" * 70, flush=True)
        print("   3. CALCULATING OFFICIAL FID SCORES", flush=True)
        print("=" * 70, flush=True)
        try:
            print("[*] Calculating Rejuvenation FID (Generated Rejuv vs Real Young Test)...", flush=True)
            rejuv_fid = float(fid_score.calculate_fid_given_paths([y_test_dir, gen_rejuv_dir], batch_size=32, device=dev, dims=2048))
            print(f"    -> Rejuvenation Low-FID: {rejuv_fid:.2f}", flush=True)
        except Exception as e:
            print(f"    [!] Error calculating Rejuvenation FID: {e}", flush=True)

        try:
            print("[*] Calculating Aging FID (Generated Aging vs Real Senescent Test)...", flush=True)
            aging_fid = float(fid_score.calculate_fid_given_paths([s_test_dir, gen_aging_dir], batch_size=32, device=dev, dims=2048))
            print(f"    -> Aging Low-FID: {aging_fid:.2f}", flush=True)
        except Exception as e:
            print(f"    [!] Error calculating Aging FID: {e}", flush=True)

    # Save Master Summary CSV
    df_summary = pd.DataFrame([
        {
            'Direction': 'Rejuvenation',
            'Strength': 0.65,
            'CFG': 3.5,
            'Target_Passage': 3.5,
            'N_Samples': len(s_files),
            'FID': round(rejuv_fid, 2) if rejuv_fid is not None else 'N/A',
            'Input_Passage_Mean': round(float(real_s_preds.mean()), 2),
            'Output_Passage_Mean': round(float(p_rejuv.mean()), 2),
            'Delta_Passage': round(delta_p_rejuv, 2),
            'Success_Rate_pct': round(rejuv_success_pct, 1),
            'Input_Brightness': round(float(real_s_bright.mean()), 4),
            'Output_Brightness': round(float(b_rejuv.mean()), 4),
            'Brightness_Shift': round(bright_shift_rejuv, 4),
        },
        {
            'Direction': 'Aging',
            'Strength': 0.75,
            'CFG': 4.0,
            'Target_Passage': 15.5,
            'N_Samples': len(y_files),
            'FID': round(aging_fid, 2) if aging_fid is not None else 'N/A',
            'Input_Passage_Mean': round(float(real_y_preds.mean()), 2),
            'Output_Passage_Mean': round(float(p_aging.mean()), 2),
            'Delta_Passage': round(delta_p_aging, 2),
            'Success_Rate_pct': round(aging_success_pct, 1),
            'Input_Brightness': round(float(real_y_bright.mean()), 4),
            'Output_Brightness': round(float(b_aging.mean()), 4),
            'Brightness_Shift': round(bright_shift_aging, 4),
        }
    ])
    csv_path = os.path.join(out_dir, 'v16_stage2_lowfid_summary.csv')
    df_summary.to_csv(csv_path, index=False)
    print(f"\n[*] Canonical Low-FID summary saved to: {csv_path}", flush=True)
    print("[*] All completed successfully!", flush=True)


if __name__ == '__main__':
    main()
