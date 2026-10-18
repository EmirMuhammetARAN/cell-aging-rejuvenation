"""
LDM v14 Comprehensive Benchmark & Comparative Evaluation
=========================================================
Evaluates LDM v14 (Mask-Guided Region-Focused Fine-Tuning) against the LDM v12 baseline
across both Low-FID and High-Accuracy regimes on the official test set (100 DDIM steps):
  - Aging (Young -> Senescent): N = 327 test crops, dilation = 24 px
  - Rejuvenation (Senescent -> Young): N = 315 test crops, dilation = 8 px

Evaluated Dimensions:
  1. Distribution Fidelity: Inception-v3 FID (vs real ground-truth test distributions)
  2. Phenotypic Conversion: ResNet-18 Target Accuracy & Mean Confidence (%)
  3. Spatial Preservation: Background MSE (outside original cell footprint M_0)
  4. Phenotypic Shift: Within-Cell Footprint Pixel MSE (inside M_0)
  5. Boundary Integrity: Gradient-Magnitude Difference across dilation transition ring
  6. Visual Side-by-Side: [Input | Mask | v12 Standard | v12 Masked | v14 Standard | v14 Masked]

Usage:
    python src/inference/ldm/eval_v14_benchmark.py --regime both --steps 100
"""

import os
import sys

# Ensure UTF-8 output on Windows terminal
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
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v14 import CellLDM_v14
from models.classifier.classifier import Classifier


def extract_fov_id(filename: str) -> str:
    m = re.match(r'^(.*)_[0-9]+\.(?:jpg|png)$', filename)
    return m.group(1) if m else filename


def compute_boundary_gradient_diff_gpu(orig_t: torch.Tensor,
                                       out_t: torch.Tensor,
                                       mask_t: torch.Tensor,
                                       dilation_px: int) -> float:
    """Measures gradient magnitude difference across the dilation transition ring."""
    m_4d = mask_t if mask_t.dim() == 4 else mask_t.unsqueeze(0).float()
    pad = dilation_px + 2
    m_dil_outer = F.max_pool2d(m_4d, kernel_size=2 * pad + 1, stride=1, padding=pad)
    d_inner = max(0, dilation_px - 2)
    m_dil_inner = F.max_pool2d(m_4d, kernel_size=2 * d_inner + 1, stride=1, padding=d_inner) if d_inner > 0 else m_4d
    ring = (m_dil_outer - m_dil_inner).clamp(0.0, 1.0)

    if ring.sum() < 1.0:
        return 0.0

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=orig_t.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=torch.float32, device=orig_t.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    gx_in = F.conv2d(orig_t, sobel_x, padding=1, groups=3)
    gy_in = F.conv2d(orig_t, sobel_y, padding=1, groups=3)
    mag_in = torch.sqrt(gx_in ** 2 + gy_in ** 2 + 1e-8)

    gx_out = F.conv2d(out_t, sobel_x, padding=1, groups=3)
    gy_out = F.conv2d(out_t, sobel_y, padding=1, groups=3)
    mag_out = torch.sqrt(gx_out ** 2 + gy_out ** 2 + 1e-8)

    diff = torch.abs(mag_out - mag_in)
    val = (diff * ring).sum() / (ring.sum() * 3.0)
    return float(val.item())


def load_v14_model(device: str, checkpoint_path: str):
    print(f"[*] Initializing CellLDM_v14 architecture...")
    model = CellLDM_v14(num_classes=2, lpips_weight=0.0)
    model.scaling_factor = 0.18215
    model.to(device, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema()

    print(f"[*] Loading v14 checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    for key in ['unet_state_dict', 'ema_unet_state_dict']:
        if key in ckpt and 'class_embedding.weight' in ckpt[key]:
            w = ckpt[key]['class_embedding.weight']
            if w.shape[0] == 2:
                pad = torch.zeros(1, w.shape[1], device=w.device)
                ckpt[key]['class_embedding.weight'] = torch.cat([w, pad], dim=0)

    model.unet.load_state_dict(ckpt['unet_state_dict'])
    if 'ema_unet_state_dict' in ckpt:
        model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
    print(f"[OK] Loaded LDM v14 weights from epoch {ckpt.get('epoch', '?')}")
    del ckpt
    gc.collect()
    model.eval()
    return model


def evaluate_auxiliary_classifier(img_dir: str, target_class: int, classifier, device):
    cls_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
    correct = 0
    total = 0
    confidences = []

    with torch.no_grad():
        for i in range(0, len(files), 32):
            batch_fns = files[i:i + 32]
            imgs = [cls_transform(Image.open(os.path.join(img_dir, fn)).convert('RGB')) for fn in batch_fns]
            batch_tensor = torch.stack(imgs, dim=0).to(device)

            logits = classifier(batch_tensor)
            probs = F.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)
            target_probs = probs[:, target_class]

            correct += (preds == target_class).sum().item()
            total += len(batch_fns)
            confidences.extend(target_probs.cpu().tolist())

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    mean_conf = float(np.mean(confidences)) * 100.0 if len(confidences) > 0 else 0.0
    return acc, mean_conf, total


def compute_spatial_mse_metrics(source_dir: str, gen_dir: str, mrcnn_masks: dict, dilation_px: int, device: str):
    """Computes mean Background MSE, Cell Footprint MSE, and Boundary Gradient Difference."""
    files = sorted([f for f in os.listdir(gen_dir) if f in mrcnn_masks and mrcnn_masks[f].sum() > 0])
    bg_mses = []
    cell_mses = []
    grad_diffs = []

    to_tensor = transforms.ToTensor()

    for fn in files:
        orig_pil = Image.open(os.path.join(source_dir, fn)).convert('RGB')
        gen_pil = Image.open(os.path.join(gen_dir, fn)).convert('RGB')

        orig_t = to_tensor(orig_pil).unsqueeze(0).to(device)
        gen_t = to_tensor(gen_pil).unsqueeze(0).to(device)

        m_orig = torch.from_numpy(mrcnn_masks[fn].astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        if m_orig.shape[2:] != orig_t.shape[2:]:
            m_orig = F.interpolate(m_orig, size=orig_t.shape[2:], mode='nearest')

        m_bg = (1.0 - m_orig).clamp(0.0, 1.0)

        # Background MSE (outside initial cell footprint)
        diff_sq = (gen_t - orig_t) ** 2
        bg_mse = float(((diff_sq * m_bg).sum() / (m_bg.sum() * 3.0)).item())
        cell_mse = float(((diff_sq * m_orig).sum() / (m_orig.sum() * 3.0)).item())

        # Boundary Gradient Difference
        grad_diff = compute_boundary_gradient_diff_gpu(orig_t, gen_t, m_orig, dilation_px)

        bg_mses.append(bg_mse)
        cell_mses.append(cell_mse)
        grad_diffs.append(grad_diff)

    return float(np.mean(bg_mses)), float(np.mean(cell_mses)), float(np.mean(grad_diffs))


def main():
    parser = argparse.ArgumentParser(description="LDM v14 Comprehensive Benchmark")
    parser.add_argument('--regime', type=str, default='both', choices=['low_fid', 'high_acc', 'both'],
                        help='Evaluation regime (low_fid, high_acc, or both)')
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of DDIM steps (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for generation')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v14_mask_guided_last.pt'),
                        help='LDM v14 checkpoint path')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(root_dir, 'results', 'v14_evaluation'),
                        help='Base output directory')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip generation and run metrics on existing images')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_out_dir = os.path.join(args.output_dir, f"{args.steps}steps")
    os.makedirs(base_out_dir, exist_ok=True)

    print("\n" + "=" * 90)
    print("       LDM v14 (MASK-GUIDED FINE-TUNED) BENCHMARK & COMPARATIVE EVALUATION       ")
    print("=" * 90)
    print(f"  - Checkpoint:          {args.checkpoint}")
    print(f"  - Target DDIM Steps:   {args.steps}")
    print(f"  - Output Directory:    {base_out_dir}")
    print(f"  - Device:              {device.upper()} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print("=" * 90 + "\n")

    all_regimes = {
        'low_fid': {
            'aging': {'strength': 0.75, 'cfg': 4.0, 'dilation': 24},
            'rejuvenation': {'strength': 0.65, 'cfg': 3.5, 'dilation': 8}
        },
        'high_acc': {
            'aging': {'strength': 0.80, 'cfg': 6.0, 'dilation': 24},
            'rejuvenation': {'strength': 0.70, 'cfg': 5.0, 'dilation': 8}
        }
    }

    regimes = all_regimes if args.regime == 'both' else {args.regime: all_regimes[args.regime]}

    source_paths = {
        'aging': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young'),
        'rejuvenation': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
    }
    ref_paths = {
        'aging': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent'),
        'rejuvenation': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    }
    mask_paths = {
        'aging': os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_young.npy'),
        'rejuvenation': os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_senescent.npy')
    }
    target_labels = {
        'aging': 1,        # LDM senescent class
        'rejuvenation': 0  # LDM young class
    }
    # ResNet-18 classifier target class: Senescent = 0, Young = 1
    cls_targets = {
        'aging': 0,
        'rejuvenation': 1
    }

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # ---------------------------------------------------------
    # STEP 1: Generation with LDM v14 (Standard and Masked)
    # ---------------------------------------------------------
    if not args.eval_only:
        model = load_v14_model(device, args.checkpoint)

        for regime_name, configs in regimes.items():
            regime_dir = os.path.join(base_out_dir, regime_name)
            os.makedirs(regime_dir, exist_ok=True)

            for direction, cfg in configs.items():
                print(f"\n[>>>] Generating LDM v14 for {regime_name.upper()} - {direction.upper()} "
                      f"(strength={cfg['strength']}, CFG={cfg['cfg']}, dilation={cfg['dilation']} px, {args.steps} steps)...")

                source_dir = source_paths[direction]
                target_label_val = target_labels[direction]
                mrcnn_masks = np.load(mask_paths[direction], allow_pickle=True).item()

                all_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
                valid_files = [f for f in all_files if f in mrcnn_masks and mrcnn_masks[f].sum() > 0]
                print(f"      Valid single-cell cohort: {len(valid_files)} crops")

                std_dir = os.path.join(regime_dir, f"{direction}_v14_standard")
                msk_dir = os.path.join(regime_dir, f"{direction}_v14_masked_{cfg['dilation']}px")
                os.makedirs(std_dir, exist_ok=True)
                os.makedirs(msk_dir, exist_ok=True)

                existing_std = set(os.listdir(std_dir))
                existing_msk = set(os.listdir(msk_dir))
                pending_files = [f for f in valid_files if f not in existing_std or f not in existing_msk]

                if len(pending_files) == 0:
                    print(f"      [OK] All {len(valid_files)} crops already generated on disk.")
                    continue

                pbar = tqdm(range(0, len(pending_files), args.batch_size), desc=f"v14-{regime_name}-{direction}")
                for start_idx in pbar:
                    batch_fns = pending_files[start_idx:start_idx + args.batch_size]
                    batch_imgs_list = []
                    batch_masks_list = []

                    for fn in batch_fns:
                        pil_img = Image.open(os.path.join(source_dir, fn)).convert('RGB')
                        batch_imgs_list.append(transform_norm(pil_img))

                        mask_np = mrcnn_masks[fn].astype(np.float32)
                        mask_t = torch.from_numpy(mask_np).unsqueeze(0)
                        if mask_t.shape[1] != 512 or mask_t.shape[2] != 512:
                            mask_t = F.interpolate(mask_t.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
                        batch_masks_list.append(mask_t)

                    batch_imgs = torch.stack(batch_imgs_list, dim=0).to(device, memory_format=torch.channels_last)
                    batch_masks = torch.stack(batch_masks_list, dim=0).to(device)
                    batch_lbls = torch.full((len(batch_fns),), target_label_val, dtype=torch.long, device=device)

                    # Exact shared initial noise per cell for paired control
                    torch.manual_seed(1000 + start_idx)
                    init_noise = torch.randn((len(batch_fns), 4, 64, 64), device=device, dtype=torch.float32)

                    # 1. LDM v14 Standard (Unmasked) Translation
                    with torch.no_grad():
                        out_std = model.translate(
                            batch_imgs, batch_lbls,
                            strength=cfg['strength'], num_steps=args.steps,
                            use_ema=True, guidance_scale=cfg['cfg'],
                            noise=init_noise
                        )

                    # 2. LDM v14 Mask-Guided Translation
                    with torch.no_grad():
                        out_msk = model.translate_masked(
                            batch_imgs, batch_lbls, cell_mask=batch_masks,
                            strength=cfg['strength'], num_steps=args.steps,
                            use_ema=True, guidance_scale=cfg['cfg'],
                            dilation_px=cfg['dilation'],
                            noise=init_noise
                        )

                    for idx_b, fn in enumerate(batch_fns):
                        save_image(out_std[idx_b], os.path.join(std_dir, fn))
                        save_image(out_msk[idx_b], os.path.join(msk_dir, fn))

                    if device == 'cuda':
                        torch.cuda.empty_cache()

                print(f"      [OK] Completed generation for {regime_name} {direction}.")

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # ---------------------------------------------------------
    # STEP 2: Comprehensive Quantitative Metrics
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("                COMPUTING QUANTITATIVE BENCHMARK METRICS                ")
    print("=" * 90)

    classifier = Classifier(output_size=2)
    cls_ckpt = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device))
    classifier.to(device)
    classifier.eval()
    print("[OK] Auxiliary ResNet-18 classifier loaded.")

    records = []

    # Path to LDM v12 baseline summary for paired comparison
    v12_summary_csv = os.path.join(root_dir, 'results', 'ablation_masked_vs_standard',
                                   f'strictly_paired_{args.steps}steps', 'strictly_paired_summary.csv')
    df_v12 = pd.read_csv(v12_summary_csv) if os.path.exists(v12_summary_csv) else None

    for regime_name, configs in regimes.items():
        regime_dir = os.path.join(base_out_dir, regime_name)
        print(f"\n>>> Evaluating Regime: {regime_name.upper()}")

        for direction, cfg in configs.items():
            d = cfg['dilation']
            std_dir = os.path.join(regime_dir, f"{direction}_v14_standard")
            msk_dir = os.path.join(regime_dir, f"{direction}_v14_masked_{d}px")
            ref_dir = ref_paths[direction]
            source_dir = source_paths[direction]
            mrcnn_masks = np.load(mask_paths[direction], allow_pickle=True).item()
            target_class = cls_targets[direction]

            # 1. FID
            print(f"  [*] Computing FID for {direction.upper()} (v14 Standard)...")
            fid_std = float(fid_score.calculate_fid_given_paths([std_dir, ref_dir], batch_size=32, device=device, dims=2048))
            print(f"      -> FID (v14 Standard): {fid_std:.2f}")

            print(f"  [*] Computing FID for {direction.upper()} (v14 Masked {d}px)...")
            fid_msk = float(fid_score.calculate_fid_given_paths([msk_dir, ref_dir], batch_size=32, device=device, dims=2048))
            print(f"      -> FID (v14 Masked):   {fid_msk:.2f}")

            # 2. ResNet-18 Classifier Metrics
            print(f"  [*] Computing ResNet-18 metrics...")
            acc_std, conf_std, tot_std = evaluate_auxiliary_classifier(std_dir, target_class, classifier, device)
            acc_msk, conf_msk, tot_msk = evaluate_auxiliary_classifier(msk_dir, target_class, classifier, device)
            print(f"      -> v14 Standard: Acc = {acc_std:.2f}%, Conf = {conf_std:.2f}% (N={tot_std})")
            print(f"      -> v14 Masked:   Acc = {acc_msk:.2f}%, Conf = {conf_msk:.2f}% (N={tot_msk})")

            # 3. Spatial MSE & Gradient Continuity
            print(f"  [*] Computing spatial MSE & boundary gradient differences...")
            bg_mse_std, cell_mse_std, grad_std = compute_spatial_mse_metrics(source_dir, std_dir, mrcnn_masks, d, device)
            bg_mse_msk, cell_mse_msk, grad_msk = compute_spatial_mse_metrics(source_dir, msk_dir, mrcnn_masks, d, device)
            print(f"      -> v14 Standard: BG MSE = {bg_mse_std:.6f}, Cell MSE = {cell_mse_std:.6f}, Grad Diff = {grad_std:.4f}")
            print(f"      -> v14 Masked:   BG MSE = {bg_mse_msk:.6f}, Cell MSE = {cell_mse_msk:.6f}, Grad Diff = {grad_msk:.4f}")

            # Extract v12 metrics from baseline if available
            v12_row = None
            if df_v12 is not None:
                matches = df_v12[(df_v12['Regime'] == regime_name) & (df_v12['Direction'] == direction)]
                if len(matches) > 0:
                    v12_row = matches.iloc[0]

            rec = {
                'Regime': regime_name,
                'Direction': direction,
                'Strength': cfg['strength'],
                'CFG': cfg['cfg'],
                'Dilation_px': d,
                # v14 Metrics
                'FID_v14_Standard': fid_std,
                'FID_v14_Masked': fid_msk,
                'Acc_v14_Standard_pct': acc_std,
                'Acc_v14_Masked_pct': acc_msk,
                'Conf_v14_Standard_pct': conf_std,
                'Conf_v14_Masked_pct': conf_msk,
                'BG_MSE_v14_Standard': bg_mse_std,
                'BG_MSE_v14_Masked': bg_mse_msk,
                'Cell_MSE_v14_Standard': cell_mse_std,
                'Cell_MSE_v14_Masked': cell_mse_msk,
                'Grad_Diff_v14_Standard': grad_std,
                'Grad_Diff_v14_Masked': grad_msk,
                'N_Crops': tot_std
            }

            if v12_row is not None:
                rec['FID_v12_Standard'] = float(v12_row['FID_Standard'])
                rec['FID_v12_Masked'] = float(v12_row['FID_Masked'])
                rec['Acc_v12_Standard_pct'] = float(v12_row['Acc_Standard_pct'])
                rec['Acc_v12_Masked_pct'] = float(v12_row['Acc_Masked_pct'])

            records.append(rec)

    # Save summary dataframe
    df_out = pd.DataFrame(records)
    csv_path = os.path.join(base_out_dir, 'v14_benchmark_summary.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"\n[OK] Benchmark CSV saved to: {csv_path}")

    # ---------------------------------------------------------
    # STEP 3: Generate Visual Comparison Grids
    # ---------------------------------------------------------
    print("\n[*] Generating qualitative visual comparison panels...")
    for regime_name, configs in regimes.items():
        regime_dir = os.path.join(base_out_dir, regime_name)
        v12_regime_dir = os.path.join(root_dir, 'results', 'ablation_masked_vs_standard',
                                      f'strictly_paired_{args.steps}steps', regime_name)

        for direction, cfg in configs.items():
            d = cfg['dilation']
            source_dir = source_paths[direction]
            mrcnn_masks = np.load(mask_paths[direction], allow_pickle=True).item()

            std_dir_v14 = os.path.join(regime_dir, f"{direction}_v14_standard")
            msk_dir_v14 = os.path.join(regime_dir, f"{direction}_v14_masked_{d}px")

            std_dir_v12 = os.path.join(v12_regime_dir, f"{direction}_standard") if os.path.exists(v12_regime_dir) else None
            msk_dir_v12 = os.path.join(v12_regime_dir, f"{direction}_masked_{d}px") if os.path.exists(v12_regime_dir) else None

            sample_files = sorted(os.listdir(std_dir_v14))[:4]
            grid_rows = []

            to_tensor = transforms.ToTensor()

            for fn in sample_files:
                in_img = to_tensor(Image.open(os.path.join(source_dir, fn)).convert('RGB'))
                m_arr = mrcnn_masks.get(fn, np.zeros((512, 512)))
                m_t = torch.from_numpy(m_arr.astype(np.float32)).unsqueeze(0).repeat(3, 1, 1)

                # Mask overlay (red contour/tint on input)
                mask_overlay = in_img.clone()
                mask_overlay[0, m_t[0] > 0.5] = 1.0  # Highlight red

                v14_std = to_tensor(Image.open(os.path.join(std_dir_v14, fn)).convert('RGB'))
                v14_msk = to_tensor(Image.open(os.path.join(msk_dir_v14, fn)).convert('RGB'))

                row_items = [in_img, mask_overlay]

                if std_dir_v12 and os.path.exists(os.path.join(std_dir_v12, fn)):
                    v12_std = to_tensor(Image.open(os.path.join(std_dir_v12, fn)).convert('RGB'))
                    row_items.append(v12_std)
                if msk_dir_v12 and os.path.exists(os.path.join(msk_dir_v12, fn)):
                    v12_msk = to_tensor(Image.open(os.path.join(msk_dir_v12, fn)).convert('RGB'))
                    row_items.append(v12_msk)

                row_items.extend([v14_std, v14_msk])
                grid_rows.append(torch.cat(row_items, dim=2))

            if len(grid_rows) > 0:
                full_grid = torch.cat(grid_rows, dim=1)
                grid_path = os.path.join(base_out_dir, f"figure_comparison_{regime_name}_{direction}.png")
                save_image(full_grid, grid_path)
                print(f"      ✓ Comparison panel saved: {grid_path}")

    # ---------------------------------------------------------
    # STEP 4: Scientific Comparative Report
    # ---------------------------------------------------------
    report_path = os.path.join(base_out_dir, 'v14_vs_v12_comparison_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 125 + "\n")
        f.write("LDM v14 (MASK-GUIDED REGION-FOCUSED FINE-TUNING) VS. LDM v12 BASELINE BENCHMARK\n")
        f.write("=" * 125 + "\n\n")
        f.write("Experimental Setup:\n")
        f.write(f"  - Evaluated Checkpoint:     {args.checkpoint}\n")
        f.write(f"  - DDIM Sampling Steps:      {args.steps}\n")
        f.write("  - Auxiliary Classifier:     ResNet-18 (classifier_v2.pth)\n")
        f.write("  - Spatial Dilation Margin:  Aging = 24 px, Rejuvenation = 8 px\n")
        f.write("  - Evaluated Cohorts:        N = 327 Aging crops, N = 315 Rejuvenation crops\n\n")

        f.write("-" * 125 + "\n")
        f.write(f"{'REGIME':<9} | {'TASK':<12} | {'CONFIG':<14} | {'METRIC':<24} | {'v12 STD':<10} | {'v12 MSK':<10} | {'v14 STD':<10} | {'v14 MSK':<10} | {'BEST'}\n")
        f.write("-" * 125 + "\n")

        for r in records:
            reg = r['Regime'].upper()
            task = r['Direction'].capitalize()
            cfg_str = f"s={r['Strength']},cfg={r['CFG']}"

            v12_s_fid = f"{r.get('FID_v12_Standard', 0.0):.2f}" if 'FID_v12_Standard' in r else "N/A"
            v12_m_fid = f"{r.get('FID_v12_Masked', 0.0):.2f}" if 'FID_v12_Masked' in r else "N/A"
            v14_s_fid = f"{r['FID_v14_Standard']:.2f}"
            v14_m_fid = f"{r['FID_v14_Masked']:.2f}"

            f.write(f"{reg:<9} | {task:<12} | {cfg_str:<14} | {'FID (lower is best)':<24} | {v12_s_fid:<10} | {v12_m_fid:<10} | {v14_s_fid:<10} | {v14_m_fid:<10} |\n")

            v12_s_acc = f"{r.get('Acc_v12_Standard_pct', 0.0):.1f}%" if 'Acc_v12_Standard_pct' in r else "N/A"
            v12_m_acc = f"{r.get('Acc_v12_Masked_pct', 0.0):.1f}%" if 'Acc_v12_Masked_pct' in r else "N/A"
            v14_s_acc = f"{r['Acc_v14_Standard_pct']:.1f}%"
            v14_m_acc = f"{r['Acc_v14_Masked_pct']:.1f}%"

            f.write(f"{'':<9} | {'':<12} | {'':<14} | {'ResNet ACC (%)':<24} | {v12_s_acc:<10} | {v12_m_acc:<10} | {v14_s_acc:<10} | {v14_m_acc:<10} |\n")

            f.write(f"{'':<9} | {'':<12} | {'':<14} | {'Background MSE':<24} | {'-':<10} | {'-':<10} | {r['BG_MSE_v14_Standard']:<10.5f} | {r['BG_MSE_v14_Masked']:<10.5f} |\n")
            f.write(f"{'':<9} | {'':<12} | {'':<14} | {'Cell Footprint MSE':<24} | {'-':<10} | {'-':<10} | {r['Cell_MSE_v14_Standard']:<10.5f} | {r['Cell_MSE_v14_Masked']:<10.5f} |\n")
            f.write(f"{'':<9} | {'':<12} | {'':<14} | {'Boundary Grad Diff':<24} | {'-':<10} | {'-':<10} | {r['Grad_Diff_v14_Standard']:<10.4f} | {r['Grad_Diff_v14_Masked']:<10.4f} |\n")
            f.write("-" * 125 + "\n")

    print(f"\n[OK] Comparative report generated at: {report_path}\n")
    with open(report_path, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == '__main__':
    main()
