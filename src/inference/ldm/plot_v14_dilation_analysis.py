"""
Generates Compact Publication Figures: LDM v12 (Standard) vs. LDM v14 (Mask-Guided)
=====================================================================================
Produces a clean 4-column publication-grade comparison:
  Column 1: Original Input (x_0)
  Column 2: Mask R-CNN Footprint & Dilation Envelope (24 px Aging / 8 px Rejuvenation)
  Column 3: LDM v12 Standard Baseline (Full Canvas Translation)
  Column 4: LDM v14 Mask-Guided (Proposed Final Model)

Outputs:
  - results/v14_evaluation/figure_v12_vs_v14_compact_low_fid.png
  - results/v14_evaluation/figure_v12_vs_v14_compact_high_acc.png
  - results/v14_evaluation/figure_v12_vs_v14_compact_master.png

Usage:
  python src/inference/ldm/plot_v14_dilation_analysis.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

out_dir = os.path.join(root_dir, 'results', 'v14_evaluation')
os.makedirs(out_dir, exist_ok=True)


def build_mask_envelope_overlay(img_pil: Image.Image, mask_np: np.ndarray, dilation_px: int) -> np.ndarray:
    """Overlays the original cell footprint (cyan) and dilation expansion envelope (amber contour)."""
    img_arr = np.array(img_pil).astype(np.float32) / 255.0
    h, w, _ = img_arr.shape

    m_orig = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    if m_orig.shape[2:] != (h, w):
        m_orig = F.interpolate(m_orig, size=(h, w), mode='nearest')

    m_dil = F.max_pool2d(m_orig, kernel_size=2 * dilation_px + 1, stride=1, padding=dilation_px)

    orig_mask = m_orig.squeeze().numpy() > 0.5
    dil_mask = m_dil.squeeze().numpy() > 0.5
    ring_mask = dil_mask & (~orig_mask)

    overlay = img_arr.copy()

    # Tint cell footprint in subtle cyan/green
    overlay[orig_mask, 0] = overlay[orig_mask, 0] * 0.7 + 0.1
    overlay[orig_mask, 1] = overlay[orig_mask, 1] * 0.7 + 0.3
    overlay[orig_mask, 2] = overlay[orig_mask, 2] * 0.7 + 0.3

    # Tint dilation envelope in subtle amber/orange
    overlay[ring_mask, 0] = overlay[ring_mask, 0] * 0.7 + 0.3
    overlay[ring_mask, 1] = overlay[ring_mask, 1] * 0.7 + 0.2
    overlay[ring_mask, 2] = overlay[ring_mask, 2] * 0.7 + 0.05

    return np.clip(overlay, 0.0, 1.0)


def generate_regime_figure(regime: str, n_samples_per_task: int = 3, out_filename: str = None):
    v12_dir = os.path.join(root_dir, 'results', 'ablation_masked_vs_standard', 'strictly_paired_100steps', regime)
    v14_dir = os.path.join(root_dir, 'results', 'v14_evaluation', '100steps', regime)

    src_dirs = {
        'aging': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young'),
        'rejuvenation': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
    }

    mask_files = {
        'aging': np.load(os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_young.npy'), allow_pickle=True).item(),
        'rejuvenation': np.load(os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_senescent.npy'), allow_pickle=True).item()
    }

    dilations = {'aging': 24, 'rejuvenation': 8}

    selected_files = {}
    for task in ['aging', 'rejuvenation']:
        v14_task_dir = os.path.join(v14_dir, f"{task}_v14_masked_{dilations[task]}px")
        v12_task_dir = os.path.join(v12_dir, f"{task}_standard")
        m_dict = mask_files[task]

        candidates = sorted([
            f for f in os.listdir(v14_task_dir)
            if f in m_dict and m_dict[f].sum() > 3000 and os.path.exists(os.path.join(v12_task_dir, f))
        ])

        step = max(1, len(candidates) // n_samples_per_task)
        selected_files[task] = [candidates[i * step] for i in range(n_samples_per_task)]

    total_rows = n_samples_per_task * 2
    fig, axes = plt.subplots(total_rows, 4, figsize=(16, 4.0 * total_rows), dpi=300)
    plt.subplots_adjust(wspace=0.04, hspace=0.15)

    col_titles = [
        "Original Input ($x_0$)",
        "Segmented Mask &\nGuidance Margin ($M_d$)",
        "LDM v12 Standard\n(Unconstrained Baseline)",
        "LDM v14 Mask-Guided\n(Proposed Final Model)"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, fontweight='bold', pad=12)

    row_idx = 0
    tasks_info = [
        ('aging', f"Aging (Young -> Senescent | Margin: {dilations['aging']} px)"),
        ('rejuvenation', f"Rejuvenation (Senescent -> Young | Margin: {dilations['rejuvenation']} px)")
    ]

    for task, _ in tasks_info:
        v14_task_dir = os.path.join(v14_dir, f"{task}_v14_masked_{dilations[task]}px")
        v12_task_dir = os.path.join(v12_dir, f"{task}_standard")
        m_dict = mask_files[task]
        d = dilations[task]

        for sample_i, fn in enumerate(selected_files[task]):
            orig_pil = Image.open(os.path.join(src_dirs[task], fn)).convert('RGB')
            v12_pil = Image.open(os.path.join(v12_task_dir, fn)).convert('RGB')
            v14_pil = Image.open(os.path.join(v14_task_dir, fn)).convert('RGB')
            mask_np = m_dict[fn]

            axes[row_idx, 0].imshow(orig_pil)
            axes[row_idx, 0].set_ylabel(f"{task.upper()}\nCell #{sample_i+1}", fontsize=11, fontweight='bold')

            mask_overlay = build_mask_envelope_overlay(orig_pil, mask_np, d)
            axes[row_idx, 1].imshow(mask_overlay)
            axes[row_idx, 2].imshow(v12_pil)
            axes[row_idx, 3].imshow(v14_pil)

            for col in range(4):
                axes[row_idx, col].set_xticks([])
                axes[row_idx, col].set_yticks([])

            row_idx += 1

    regime_title = "Low-FID Regime (Aging: s=0.75, CFG=4.0 | Rejuvenation: s=0.65, CFG=3.5)" if regime == 'low_fid' else "High-Accuracy Regime (Aging: s=0.80, CFG=6.0 | Rejuvenation: s=0.70, CFG=5.0)"
    fig.suptitle(f"Compact Model Benchmark: LDM v12 (Standard Baseline) vs. LDM v14 (Mask-Guided)\n{regime_title}",
                 fontsize=15, fontweight='bold', y=0.995)

    if out_filename is None:
        out_filename = f"figure_v12_vs_v14_compact_{regime}.png"

    save_path = os.path.join(out_dir, out_filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[OK] Compact figure saved to: {save_path}")
    return save_path


def generate_unified_combined_figure():
    return generate_regime_figure('low_fid', n_samples_per_task=2, out_filename="figure_v12_vs_v14_compact_master.png")


if __name__ == '__main__':
    print("Generating compact figures...")
    generate_regime_figure('low_fid', n_samples_per_task=3, out_filename="figure_v12_vs_v14_compact_low_fid.png")
    generate_regime_figure('high_acc', n_samples_per_task=3, out_filename="figure_v12_vs_v14_compact_high_acc.png")
    generate_unified_combined_figure()
    print("[DONE] All compact figures generated successfully.")
