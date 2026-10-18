"""
Master 3-Way Training Dynamics & Volatility Benchmark:
CycleGAN vs. CUT GAN vs. Latent Diffusion Model (LDM v12) - Best Seed & 5-Seed Invariance
========================================================================================
Generates:
  - figure_grand_three_way_fid.png (Full scale up to 300 with annotated spikes, zoom insets, & Best Seed callouts)
  - figure_grand_three_way_acc.png (Classifier Accuracy across epochs for all 3 models with Best Seed callouts)
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))

OUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm_v12_sweep')
os.makedirs(OUT_DIR, exist_ok=True)
ARTIFACT_DIR = r'C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7'

# 1. CycleGAN Data (from sweep_epochs_v2 & plot_fid_comparison.py)
CYCLEGAN_EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
CYCLEGAN_AGING_FID = [58.66, 62.88, 57.33, 63.72, 66.56, 68.08, 59.98, 89.22, 68.40, 56.76, 66.65, 57.68, 64.07, 62.08, 60.79, 53.70, 60.33, 272.05, 65.53, 70.54]
CYCLEGAN_REJU_FID = [69.54, 61.59, 87.94, 82.79, 66.45, 66.63, 67.41, 63.64, 65.65, 64.96, 208.81, 65.52, 59.88, 68.49, 79.30, 63.52, 143.67, 120.00, 73.90, 66.44]
CYCLEGAN_AGING_ACC = [78.90, 66.97, 65.14, 48.01, 23.24, 26.30, 44.04, 97.25, 97.25, 48.93, 39.14, 47.71, 61.77, 52.60, 54.13, 87.16, 66.97, 99.69, 35.78, 97.25]
CYCLEGAN_REJU_ACC = [77.67, 69.50, 69.81, 78.30, 73.58, 67.30, 50.63, 59.43, 73.27, 76.73, 27.67, 87.74, 82.70, 61.01, 37.11, 72.33, 55.03, 23.27, 50.63, 54.40]
CYCLEGAN_OVERALL_ACC = [78.29, 68.22, 67.44, 62.95, 48.06, 46.51, 47.29, 78.60, 85.43, 62.64, 33.49, 67.44, 72.09, 56.74, 45.74, 79.84, 61.09, 62.02, 43.10, 76.12]

# 2. CUT GAN Data (from cut_sweep_metrics.json)
CUT_JSON = os.path.join(root_dir, 'results', 'generated', 'cut_sweep', 'cut_sweep_metrics.json')
with open(CUT_JSON, 'r', encoding='utf-8') as f:
    cut_records = json.load(f)

CUT_EPOCHS = [r['epoch'] for r in cut_records]
CUT_AGING_FID = [r['aging_fid'] for r in cut_records]
CUT_REJU_FID = [r['reju_fid'] for r in cut_records]
CUT_AGING_ACC = [r['aging_acc'] for r in cut_records]
CUT_REJU_ACC = [r['reju_acc'] for r in cut_records]
CUT_OVERALL_ACC = [r['overall_acc'] for r in cut_records]

# 3. 5-Seed Stats for Final Ep 300 / Ep 200 / Ep 160
# LDM: Aging FID = 47.31 (Best Seed 2026), Rejuv FID = 53.49 (Best Seed 2026), Overall ACC = 89.15% (Best Seed 9999) / 88.84% (Seed 2026)
LDM_5SEED_AGING_FID_MEAN = 48.81
LDM_5SEED_AGING_FID_STD = 0.89
LDM_5SEED_REJU_FID_MEAN = 54.00
LDM_5SEED_REJU_FID_STD = 0.49
LDM_5SEED_OVERALL_ACC_MEAN = 88.65
LDM_5SEED_OVERALL_ACC_STD = 0.49


def main():
    # Load LDM Data (Evaluated at Best Seed 2026 across 7 checkpoints)
    ldm_json = os.path.join(root_dir, 'results', 'generated', 'ldm_v12_sweep_100steps_seed2026', 'ldm_v12_sweep_100steps_seed2026_metrics.json')
    if not os.path.exists(ldm_json):
        print(f"[!] LDM metrics file not ready yet at {ldm_json}")
        return

    with open(ldm_json, 'r', encoding='utf-8') as f:
        ldm_records = json.load(f)

    LDM_EPOCHS = [r['epoch'] for r in ldm_records]
    LDM_AGING_FID = [r['aging_fid'] for r in ldm_records]
    LDM_REJU_FID = [r['reju_fid'] for r in ldm_records]
    LDM_AGING_ACC = [r['aging_acc'] for r in ldm_records]
    LDM_REJU_ACC = [r['reju_acc'] for r in ldm_records]
    LDM_OVERALL_ACC = [r['overall_acc'] for r in ldm_records]

    plt.style.use('seaborn-v0_8-whitegrid')

    # =========================================================================
    # FIGURE 1: 3-WAY FID COMPARISON (2x1) - FULL SCALE 300 WITH INSET ZOOM
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    # ---------------------------------------------------------
    # 1a. Aging FID: Full Scale (30 to 300) + Inset Zoom (40 to 80)
    # ---------------------------------------------------------
    ax1 = axes[0]
    ax1.plot(CYCLEGAN_EPOCHS, CYCLEGAN_AGING_FID, 's--', color='#1f77b4', linewidth=1.8, alpha=0.75, label='CycleGAN Aging FID (Selected Ep 160: 53.70)')
    ax1.plot(CUT_EPOCHS, CUT_AGING_FID, 'o-', color='#2ca02c', linewidth=2.4, markersize=6, label='CUT GAN Aging FID (Selected Ep 200: 43.13)')
    ax1.plot(LDM_EPOCHS, LDM_AGING_FID, 'd-', color='#d62728', linewidth=3.0, markersize=8, label='LDM v12 Aging FID (Seed 2026: 47.31)')
    
    # 5-Seed Error Bar for LDM at Epoch 300
    ax1.errorbar([300], [LDM_5SEED_AGING_FID_MEAN], yerr=[LDM_5SEED_AGING_FID_STD], fmt='none', ecolor='#900C3F', elinewidth=2.5, capsize=6, capthick=2, label='LDM Ep 300 5-Seed Var (48.81±0.89)')

    # Annotate CycleGAN Spike at Ep 180 (FID = 272.05)
    ax1.annotate(
        'CycleGAN FID Spike\n(Ep 180: FID = 272.05)',
        xy=(180, 272.05), xytext=(65, 235),
        arrowprops=dict(facecolor='#1f77b4', edgecolor='#1f77b4', shrink=0.08, width=1.8, headwidth=8),
        fontsize=10.5, fontweight='bold', color='#0b3c5d',
        bbox=dict(boxstyle='round,pad=0.4', fc='#e8f1f5', ec='#1f77b4', lw=1.5)
    )

    # Annotate LDM Seed 2026 at Epoch 300
    ax1.annotate(
        'LDM (Seed 2026)\nFID = 47.31',
        xy=(300, 47.31), xytext=(240, 110),
        arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.6, headwidth=7),
        fontsize=10, fontweight='bold', color='#900C3F',
        bbox=dict(boxstyle='round,pad=0.35', fc='#ffebee', ec='#d62728', lw=1.3)
    )

    ax1.set_title('Aging Task: FID Trajectory Across Epochs (CycleGAN vs CUT GAN vs LDM v12)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('FID Score (Lower is Better)', fontsize=12)
    ax1.set_ylim(30, 300)
    ax1.set_xlim(0, 318)
    ax1.legend(fontsize=10.5, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax1.axvline(x=250, color='#888888', linestyle=':', linewidth=1.3, alpha=0.8)

    # Inset Axes for Aging (Zoom in 40 to 75 FID)
    ax1_ins = ax1.inset_axes([0.52, 0.44, 0.44, 0.50])
    ax1_ins.plot(CYCLEGAN_EPOCHS, CYCLEGAN_AGING_FID, 's--', color='#1f77b4', linewidth=1.5, alpha=0.6)
    ax1_ins.plot(CUT_EPOCHS, CUT_AGING_FID, 'o-', color='#2ca02c', linewidth=2.0, markersize=5)
    ax1_ins.plot(LDM_EPOCHS, LDM_AGING_FID, 'd-', color='#d62728', linewidth=2.5, markersize=6)
    ax1_ins.errorbar([300], [LDM_5SEED_AGING_FID_MEAN], yerr=[LDM_5SEED_AGING_FID_STD], fmt='none', ecolor='#900C3F', elinewidth=2, capsize=4, capthick=1.5)
    ax1_ins.axvline(x=250, color='#888888', linestyle=':', linewidth=1.0, alpha=0.7)
    ax1_ins.text(252, 73, 'Latent-L1 FT', fontsize=8, color='#555555')
    ax1_ins.set_xlim(40, 308)
    ax1_ins.set_ylim(40, 75)
    ax1_ins.set_title('Zoomed High-Fidelity Region (FID 40-75)', fontsize=9.5, fontweight='bold', color='#333333')
    ax1_ins.grid(True, linestyle=':', alpha=0.7)
    ax1.indicate_inset_zoom(ax1_ins, edgecolor="#666666", alpha=0.7)

    # ---------------------------------------------------------
    # 1b. Rejuvenation FID: Full Scale (30 to 240) + Inset Zoom (45 to 75)
    # ---------------------------------------------------------
    ax2 = axes[1]
    ax2.plot(CYCLEGAN_EPOCHS, CYCLEGAN_REJU_FID, 'x--', color='#9467bd', linewidth=1.8, alpha=0.75, label='CycleGAN Rejuv FID (Selected Ep 160: 63.52)')
    ax2.plot(CUT_EPOCHS, CUT_REJU_FID, '^-', color='#ff7f0e', linewidth=2.4, markersize=6, label='CUT GAN Rejuv FID (Selected Ep 200: 48.58)')
    ax2.plot(LDM_EPOCHS, LDM_REJU_FID, 'd-', color='#d62728', linewidth=3.0, markersize=8, label='LDM v12 Rejuv FID (Seed 2026: 53.49)')
    
    # 5-Seed Error Bar for LDM at Epoch 300
    ax2.errorbar([300], [LDM_5SEED_REJU_FID_MEAN], yerr=[LDM_5SEED_REJU_FID_STD], fmt='none', ecolor='#900C3F', elinewidth=2.5, capsize=6, capthick=2, label='LDM Ep 300 5-Seed Var (54.00±0.49)')

    # Annotate CycleGAN Spike at Ep 110 (FID = 208.81)
    ax2.annotate(
        'CycleGAN Divergence Spike\n(Ep 110: FID = 208.81)',
        xy=(110, 208.81), xytext=(55, 175),
        arrowprops=dict(facecolor='#9467bd', edgecolor='#9467bd', shrink=0.08, width=1.8, headwidth=8),
        fontsize=10.5, fontweight='bold', color='#4a148c',
        bbox=dict(boxstyle='round,pad=0.4', fc='#f3e5f5', ec='#9467bd', lw=1.5)
    )

    # Annotate CycleGAN Ep 170 (FID = 143.67)
    ax2.annotate(
        'Ep 170: FID = 143.67',
        xy=(170, 143.67), xytext=(185, 155),
        arrowprops=dict(facecolor='#9467bd', edgecolor='#9467bd', shrink=0.08, width=1.2, headwidth=6),
        fontsize=9.5, color='#4a148c',
        bbox=dict(boxstyle='round,pad=0.3', fc='#f3e5f5', ec='#9467bd', lw=1.2)
    )

    # Annotate LDM Seed 2026 at Epoch 300
    ax2.annotate(
        'LDM (Seed 2026)\nFID = 53.49',
        xy=(300, 53.49), xytext=(240, 105),
        arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.6, headwidth=7),
        fontsize=10, fontweight='bold', color='#900C3F',
        bbox=dict(boxstyle='round,pad=0.35', fc='#ffebee', ec='#d62728', lw=1.3)
    )

    ax2.axvline(x=250, color='#888888', linestyle=':', linewidth=1.3, alpha=0.8)
    ax2.text(252, 225, 'Latent-L1 FT\n(Total Ep 250-300)', fontsize=8.5, color='#444444', va='top')

    ax2.set_title('Rejuvenation Task: FID Trajectory Across Epochs (CycleGAN vs CUT GAN vs LDM v12)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Progression (Epochs)', fontsize=12)
    ax2.set_ylabel('FID Score (Lower is Better)', fontsize=12)
    ax2.set_ylim(30, 240)
    ax2.set_xlim(0, 318)
    ax2.legend(fontsize=10.5, loc='upper left')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Inset Axes for Rejuvenation (Zoom in 45 to 75 FID)
    ax2_ins = ax2.inset_axes([0.48, 0.45, 0.48, 0.48])
    ax2_ins.plot(CYCLEGAN_EPOCHS, CYCLEGAN_REJU_FID, 'x--', color='#9467bd', linewidth=1.5, alpha=0.6)
    ax2_ins.plot(CUT_EPOCHS, CUT_REJU_FID, '^-', color='#ff7f0e', linewidth=2.0, markersize=5)
    ax2_ins.plot(LDM_EPOCHS, LDM_REJU_FID, 'd-', color='#d62728', linewidth=2.5, markersize=6)
    ax2_ins.errorbar([300], [LDM_5SEED_REJU_FID_MEAN], yerr=[LDM_5SEED_REJU_FID_STD], fmt='none', ecolor='#900C3F', elinewidth=2, capsize=4, capthick=1.5)
    ax2_ins.axvline(x=250, color='#888888', linestyle=':', linewidth=1.0, alpha=0.7)
    ax2_ins.set_xlim(40, 308)
    ax2_ins.set_ylim(45, 75)
    ax2_ins.set_title('Zoomed High-Fidelity Region (FID 45-75)', fontsize=9.5, fontweight='bold', color='#333333')
    ax2_ins.grid(True, linestyle=':', alpha=0.7)
    ax2.indicate_inset_zoom(ax2_ins, edgecolor="#666666", alpha=0.7)

    fig.text(0.5, 0.008,
             'Note: Checkpoint metrics were calculated on the same 645-image evaluation cohort (327 young, 318 senescent) used for checkpoint and inference-setting comparisons.\n'
             'The cohort was excluded from model training and pseudo-labeling but was not an untouched final test set. LDM trajectory points use inference seed 2026.',
             ha='center', fontsize=8.6, style='italic', color='#333333')

    plt.tight_layout(rect=[0, 0.035, 1, 1])
    fid_fig_path = os.path.join(OUT_DIR, 'figure_grand_three_way_fid.png')
    plt.savefig(fid_fig_path, dpi=300)
    plt.close()
    print(f"[OK] Saved full-scale 3-way FID figure to {fid_fig_path}")

    # ==========================================
    # FIGURE 2: 3-WAY ACCURACY COMPARISON (3x1)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    # 2a. Aging ACC
    axes[0].plot(CYCLEGAN_EPOCHS, CYCLEGAN_AGING_ACC, 's--', color='#1f77b4', linewidth=1.8, alpha=0.7, label='CycleGAN Aging ACC (Ep 180: 99.69% [FID Spike], Selected Ep 160: 87.16%)')
    axes[0].plot(CUT_EPOCHS, CUT_AGING_ACC, 'o-', color='#2ca02c', linewidth=2.2, label='CUT GAN Aging ACC (Selected Ep 200: 72.48%)')
    axes[0].plot(LDM_EPOCHS, LDM_AGING_ACC, 'd-', color='#d62728', linewidth=3.0, markersize=8, label='LDM v12 Aging ACC (Seed 2026, Ep 300: 89.60%)')
    axes[0].axvline(x=250, color='#888888', linestyle=':', linewidth=1.3, alpha=0.8)
    axes[0].text(252, 98, 'Latent-L1 FT\n(Total Ep 250-300)', fontsize=8, color='#555555', va='top')
    axes[0].set_title('Aging Task: Auxiliary Classifier Target-Class Prediction (%) Across Epochs', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Target Prediction (%)', fontsize=11)
    axes[0].set_ylim(15, 105)
    axes[0].legend(fontsize=9.5, loc='lower left')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # 2b. Rejuv ACC
    axes[1].plot(CYCLEGAN_EPOCHS, CYCLEGAN_REJU_ACC, 'x--', color='#9467bd', linewidth=1.8, alpha=0.7, label='CycleGAN Rejuv ACC (Selected Ep 160: 72.33%)')
    axes[1].plot(CUT_EPOCHS, CUT_REJU_ACC, '^-', color='#ff7f0e', linewidth=2.2, label='CUT GAN Rejuv ACC (Selected Ep 200: 94.65%)')
    axes[1].plot(LDM_EPOCHS, LDM_REJU_ACC, 'd-', color='#d62728', linewidth=3.0, markersize=8, label='LDM v12 Rejuv ACC (Seed 2026, Ep 300: 88.05%)')
    axes[1].axvline(x=250, color='#888888', linestyle=':', linewidth=1.3, alpha=0.8)
    axes[1].text(252, 98, 'Latent-L1 FT\n(Total Ep 250-300)', fontsize=8, color='#555555', va='top')
    axes[1].set_title('Rejuvenation Task: Auxiliary Classifier Target-Class Prediction (%) Across Epochs', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Target Prediction (%)', fontsize=11)
    axes[1].set_ylim(15, 105)
    axes[1].legend(fontsize=9.5, loc='lower left')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    # 2c. Overall ACC
    axes[2].plot(CYCLEGAN_EPOCHS, CYCLEGAN_OVERALL_ACC, 'v--', color='#7f7f7f', linewidth=1.8, alpha=0.7, label='CycleGAN Overall ACC (Selected Ep 160: 79.84%)')
    axes[2].plot(CUT_EPOCHS, CUT_OVERALL_ACC, '*-', color='#2ca02c', linewidth=2.2, label='CUT GAN Overall ACC (Selected Ep 200: 83.41%)')
    axes[2].plot(LDM_EPOCHS, LDM_OVERALL_ACC, 'd-', color='#d62728', linewidth=3.2, markersize=9, label='LDM v12 Overall ACC (Seed 2026, Ep 300: 88.84%)')
    axes[2].errorbar([300], [LDM_5SEED_OVERALL_ACC_MEAN], yerr=[LDM_5SEED_OVERALL_ACC_STD], fmt='none', ecolor='#900C3F', elinewidth=2.5, capsize=6, capthick=2, label='LDM Ep 300 5-Seed Var (88.65±0.49%)')
    axes[2].axvline(x=250, color='#888888', linestyle=':', linewidth=1.3, alpha=0.8)
    axes[2].text(252, 98, 'Latent-L1 FT\n(Total Ep 250-300)', fontsize=8, color='#555555', va='top')
    axes[2].set_title('Overall Bidirectional Auxiliary Classifier Target-Class Prediction (%) Across Epochs', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Training Progression (Epochs)', fontsize=12)
    axes[2].set_ylabel('Target Prediction (%)', fontsize=11)
    axes[2].set_ylim(25, 100)
    axes[2].set_xlim(0, 318)
    axes[2].legend(fontsize=10, loc='lower right')
    axes[2].grid(True, linestyle='--', alpha=0.6)

    fig.text(0.5, 0.008,
             'Note: Checkpoint metrics were calculated on the same 645-image evaluation cohort (327 young, 318 senescent) used for checkpoint and inference-setting comparisons.\n'
             'The cohort was excluded from model training and pseudo-labeling but was not an untouched final test set. LDM trajectory points use inference seed 2026.',
             ha='center', fontsize=8.6, style='italic', color='#333333')

    plt.tight_layout(rect=[0, 0.035, 1, 1])

    # Annotation for LDM Seed 2026 Convergence
    axes[2].annotate(
        'LDM (Seed 2026)\nOverall ACC = 88.84%',
        xy=(300, 88.84), xytext=(225, 62),
        arrowprops=dict(facecolor='#d62728', edgecolor='#d62728', shrink=0.08, width=1.6, headwidth=7),
        fontsize=10, fontweight='bold', color='#900C3F',
        bbox=dict(boxstyle='round,pad=0.35', fc='#ffebee', ec='#d62728', lw=1.3)
    )

    acc_fig_path = os.path.join(OUT_DIR, 'figure_grand_three_way_acc.png')
    plt.savefig(acc_fig_path, dpi=300)
    plt.close()
    print(f"[OK] Saved 3-way Accuracy figure to {acc_fig_path}")

    # Copy to brain artifact directory
    shutil.copy2(fid_fig_path, os.path.join(ARTIFACT_DIR, 'figure_grand_three_way_fid.png'))
    shutil.copy2(acc_fig_path, os.path.join(ARTIFACT_DIR, 'figure_grand_three_way_acc.png'))
    print("[OK] Copied figures to artifact directory.")


if __name__ == '__main__':
    main()
