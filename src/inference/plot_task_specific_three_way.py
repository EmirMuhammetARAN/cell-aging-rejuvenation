"""
Task-Specific 3-Way Trajectory Benchmark:
Aging and Rejuvenation Task Comparison Across Epochs (CycleGAN vs. CUT GAN vs. LDM v12)
Evaluated with the Best FID Random Seed (Seed 2026)
========================================================================================
Generates:
  1. figure_aging_three_way_comparison.png:
     - Panel A: Aging Task FID Trajectory (Full scale 0-300 + Inset Zoom 40-75)
     - Panel B: Aging Task Transition Accuracy (%) Across Epochs
  2. figure_rejuvenation_three_way_comparison.png:
     - Panel A: Rejuvenation Task FID Trajectory (Full scale 0-240 + Inset Zoom 45-75)
     - Panel B: Rejuvenation Task Transition Accuracy (%) Across Epochs
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

OUT_DIR = os.path.join(root_dir, 'results', 'generated', 'task_comparison')
os.makedirs(OUT_DIR, exist_ok=True)
ARTIFACT_DIR = r'C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7'

# ==========================================
# 1. CycleGAN Data (Deterministic across seeds)
# ==========================================
CYCLEGAN_EPOCHS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
CYCLEGAN_AGING_FID = [58.66, 62.88, 57.33, 63.72, 66.56, 68.08, 59.98, 89.22, 68.40, 56.76, 66.65, 57.68, 64.07, 62.08, 60.79, 53.52, 60.33, 272.05, 65.53, 70.54]
CYCLEGAN_REJU_FID = [69.54, 61.59, 87.94, 82.79, 66.45, 66.63, 67.41, 63.64, 65.65, 64.96, 208.81, 65.52, 59.88, 68.49, 79.30, 63.77, 143.67, 120.00, 73.90, 66.44]
CYCLEGAN_AGING_ACC = [78.90, 66.97, 65.14, 48.01, 23.24, 26.30, 44.04, 97.25, 97.25, 48.93, 39.14, 47.71, 61.77, 52.60, 54.13, 86.24, 66.97, 99.69, 35.78, 97.25]
CYCLEGAN_REJU_ACC = [77.67, 69.50, 69.81, 78.30, 73.58, 67.30, 50.63, 59.43, 73.27, 76.73, 27.67, 87.74, 82.70, 61.01, 37.11, 71.70, 55.03, 23.27, 50.63, 54.40]

# ==========================================
# 2. CUT GAN Data (Deterministic across seeds)
# ==========================================
CUT_JSON = os.path.join(root_dir, 'results', 'generated', 'cut_sweep', 'cut_sweep_metrics.json')
with open(CUT_JSON, 'r', encoding='utf-8') as f:
    cut_records = json.load(f)

CUT_EPOCHS = [r['epoch'] for r in cut_records]
CUT_AGING_FID = [r['aging_fid'] for r in cut_records]
CUT_REJU_FID = [r['reju_fid'] for r in cut_records]
CUT_AGING_ACC = [r['aging_acc'] for r in cut_records]
CUT_REJU_ACC = [r['reju_acc'] for r in cut_records]

# ==========================================
# 3. LDM v12 Data (Best FID Seed 2026)
# ==========================================
LDM_JSON = os.path.join(root_dir, 'results', 'generated', 'ldm_v12_sweep_100steps_seed2026', 'ldm_v12_sweep_100steps_seed2026_metrics.json')
with open(LDM_JSON, 'r', encoding='utf-8') as f:
    ldm_records = json.load(f)

LDM_EPOCHS = [r['epoch'] for r in ldm_records]
LDM_AGING_FID = [r['aging_fid'] for r in ldm_records]
LDM_REJU_FID = [r['reju_fid'] for r in ldm_records]
LDM_AGING_ACC = [r['aging_acc'] for r in ldm_records]
LDM_REJU_ACC = [r['reju_acc'] for r in ldm_records]


def plot_aging_comparison():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_fid, ax_acc) = plt.subplots(2, 1, figsize=(12, 9.5), sharex=True)

    # -------------------------------------------------------------
    # PANEL A: AGING FID TRAJECTORY (Full scale 0-300 + Inset Zoom)
    # -------------------------------------------------------------
    ax_fid.plot(CYCLEGAN_EPOCHS, CYCLEGAN_AGING_FID, 's--', color='#1f77b4', linewidth=1.8, alpha=0.8,
                label='CycleGAN (Selected Ep 160: 53.52)')
    ax_fid.plot(CUT_EPOCHS, CUT_AGING_FID, 'o-', color='#2ca02c', linewidth=2.3, markersize=6,
                label='CUT GAN (Selected Ep 200: 43.13)')
    ax_fid.plot(LDM_EPOCHS, LDM_AGING_FID, 'd-', color='#d62728', linewidth=2.8, markersize=7,
                label='LDM v12 (Seed 2026, Ep 300: 47.31)')

    # Vertical line for Latent-L1 FT
    ax_fid.axvline(x=250, color='#666666', linestyle=':', linewidth=1.3, alpha=0.8)

    ax_fid.set_title('(A) Aging Task (Young → Senescent): FID Trajectory Across Epochs', fontsize=13, fontweight='bold')
    ax_fid.set_ylabel('FID Score (Lower is Better)', fontsize=11)
    ax_fid.set_ylim(30, 295)
    ax_fid.set_xlim(0, 315)
    ax_fid.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax_fid.grid(True, linestyle='--', alpha=0.55)

    # Inset Axes for Aging High-Fidelity Region (FID 40 to 75)
    ax_ins = ax_fid.inset_axes([0.48, 0.44, 0.48, 0.50])
    ax_ins.plot(CYCLEGAN_EPOCHS, CYCLEGAN_AGING_FID, 's--', color='#1f77b4', linewidth=1.5, alpha=0.6)
    ax_ins.plot(CUT_EPOCHS, CUT_AGING_FID, 'o-', color='#2ca02c', linewidth=1.8, markersize=4.5)
    ax_ins.plot(LDM_EPOCHS, LDM_AGING_FID, 'd-', color='#d62728', linewidth=2.2, markersize=5.5)
    ax_ins.axvline(x=250, color='#666666', linestyle=':', linewidth=1.0, alpha=0.7)
    ax_ins.text(252, 72, 'Latent-L1 FT', fontsize=8, color='#555555')
    ax_ins.set_xlim(40, 308)
    ax_ins.set_ylim(40, 75)
    ax_ins.set_title('High-Fidelity Region (FID 40–75)', fontsize=9, fontweight='bold', color='#333333')
    ax_ins.grid(True, linestyle=':', alpha=0.7)
    ax_fid.indicate_inset_zoom(ax_ins, edgecolor="#888888", alpha=0.6)

    # -------------------------------------------------------------
    # PANEL B: AGING ACCURACY TRAJECTORY (Auxiliary Classifier)
    # -------------------------------------------------------------
    ax_acc.plot(CYCLEGAN_EPOCHS, CYCLEGAN_AGING_ACC, 's--', color='#1f77b4', linewidth=1.8, alpha=0.7,
                label='CycleGAN (Selected Ep 160: 86.24%)')
    ax_acc.plot(CUT_EPOCHS, CUT_AGING_ACC, 'o-', color='#2ca02c', linewidth=2.2,
                label='CUT GAN (Selected Ep 200: 72.48%)')
    ax_acc.plot(LDM_EPOCHS, LDM_AGING_ACC, 'd-', color='#d62728', linewidth=2.8, markersize=7,
                label='LDM v12 (Seed 2026, Ep 300: 89.60%)')

    ax_acc.axvline(x=250, color='#666666', linestyle=':', linewidth=1.3, alpha=0.8)
    ax_acc.text(252, 98, 'Latent-L1 FT\n(Ep 250–300)', fontsize=8.5, color='#555555', va='top')

    ax_acc.set_title('(B) Aging Task (Young → Senescent): Auxiliary Classifier Target Prediction (%)', fontsize=13, fontweight='bold')
    ax_acc.set_xlabel('Training Progression (Epochs)', fontsize=11.5)
    ax_acc.set_ylabel('Target Prediction Accuracy (%)', fontsize=11)
    ax_acc.set_ylim(15, 105)
    ax_acc.set_xlim(0, 315)
    ax_acc.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax_acc.grid(True, linestyle='--', alpha=0.55)

    plt.tight_layout()
    aging_path = os.path.join(OUT_DIR, 'figure_aging_three_way_comparison.png')
    plt.savefig(aging_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Aging 3-way figure to {aging_path}")
    shutil.copy2(aging_path, os.path.join(ARTIFACT_DIR, 'figure_aging_three_way_comparison.png'))
    print(f"[OK] Copied figure_aging_three_way_comparison.png to artifact directory.")


def plot_rejuvenation_comparison():
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax_fid, ax_acc) = plt.subplots(2, 1, figsize=(12, 9.5), sharex=True)

    # -------------------------------------------------------------
    # PANEL A: REJUVENATION FID TRAJECTORY (Full scale 0-240 + Inset Zoom)
    # -------------------------------------------------------------
    ax_fid.plot(CYCLEGAN_EPOCHS, CYCLEGAN_REJU_FID, 'x--', color='#9467bd', linewidth=1.8, alpha=0.8,
                label='CycleGAN (Selected Ep 160: 63.77)')
    ax_fid.plot(CUT_EPOCHS, CUT_REJU_FID, '^-', color='#ff7f0e', linewidth=2.3, markersize=6,
                label='CUT GAN (Selected Ep 200: 48.58)')
    ax_fid.plot(LDM_EPOCHS, LDM_REJU_FID, 'd-', color='#d62728', linewidth=2.8, markersize=7,
                label='LDM v12 (Seed 2026, Ep 300: 53.49)')

    ax_fid.axvline(x=250, color='#666666', linestyle=':', linewidth=1.3, alpha=0.8)

    ax_fid.set_title('(A) Rejuvenation Task (Senescent → Young): FID Trajectory Across Epochs', fontsize=13, fontweight='bold')
    ax_fid.set_ylabel('FID Score (Lower is Better)', fontsize=11)
    ax_fid.set_ylim(30, 230)
    ax_fid.set_xlim(0, 315)
    ax_fid.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax_fid.grid(True, linestyle='--', alpha=0.55)

    # Inset Axes for Rejuv High-Fidelity Region (FID 45 to 75)
    ax_ins = ax_fid.inset_axes([0.48, 0.44, 0.48, 0.50])
    ax_ins.plot(CYCLEGAN_EPOCHS, CYCLEGAN_REJU_FID, 'x--', color='#9467bd', linewidth=1.5, alpha=0.6)
    ax_ins.plot(CUT_EPOCHS, CUT_REJU_FID, '^-', color='#ff7f0e', linewidth=1.8, markersize=5)
    ax_ins.plot(LDM_EPOCHS, LDM_REJU_FID, 'd-', color='#d62728', linewidth=2.2, markersize=5.5)
    ax_ins.axvline(x=250, color='#666666', linestyle=':', linewidth=1.0, alpha=0.7)
    ax_ins.text(252, 72, 'Latent-L1 FT', fontsize=8, color='#555555')
    ax_ins.set_xlim(40, 308)
    ax_ins.set_ylim(45, 75)
    ax_ins.set_title('High-Fidelity Region (FID 45–75)', fontsize=9, fontweight='bold', color='#333333')
    ax_ins.grid(True, linestyle=':', alpha=0.7)
    ax_fid.indicate_inset_zoom(ax_ins, edgecolor="#888888", alpha=0.6)

    # -------------------------------------------------------------
    # PANEL B: REJUVENATION ACCURACY TRAJECTORY (Auxiliary Classifier)
    # -------------------------------------------------------------
    ax_acc.plot(CYCLEGAN_EPOCHS, CYCLEGAN_REJU_ACC, 'x--', color='#9467bd', linewidth=1.8, alpha=0.7,
                label='CycleGAN (Selected Ep 160: 71.70%)')
    ax_acc.plot(CUT_EPOCHS, CUT_REJU_ACC, '^-', color='#ff7f0e', linewidth=2.2,
                label='CUT GAN (Selected Ep 200: 94.65%)')
    ax_acc.plot(LDM_EPOCHS, LDM_REJU_ACC, 'd-', color='#d62728', linewidth=2.8, markersize=7,
                label='LDM v12 (Seed 2026, Ep 300: 88.05%)')

    ax_acc.axvline(x=250, color='#666666', linestyle=':', linewidth=1.3, alpha=0.8)
    ax_acc.text(252, 98, 'Latent-L1 FT\n(Ep 250–300)', fontsize=8.5, color='#555555', va='top')

    ax_acc.set_title('(B) Rejuvenation Task (Senescent → Young): Auxiliary Classifier Target Prediction (%)', fontsize=13, fontweight='bold')
    ax_acc.set_xlabel('Training Progression (Epochs)', fontsize=11.5)
    ax_acc.set_ylabel('Target Prediction Accuracy (%)', fontsize=11)
    ax_acc.set_ylim(15, 105)
    ax_acc.set_xlim(0, 315)
    ax_acc.legend(fontsize=10, loc='lower left', framealpha=0.9)
    ax_acc.grid(True, linestyle='--', alpha=0.55)

    plt.tight_layout()
    reju_path = os.path.join(OUT_DIR, 'figure_rejuvenation_three_way_comparison.png')
    plt.savefig(reju_path, dpi=300)
    plt.close()
    print(f"[OK] Saved Rejuvenation 3-way figure to {reju_path}")
    shutil.copy2(reju_path, os.path.join(ARTIFACT_DIR, 'figure_rejuvenation_three_way_comparison.png'))
    print(f"[OK] Copied figure_rejuvenation_three_way_comparison.png to artifact directory.")


def main():
    print("[*] Generating Task-Specific 3-Way Trajectory Figures (Best Seed 2026)...")
    plot_aging_comparison()
    plot_rejuvenation_comparison()
    print("[OK] All task-specific figures generated successfully!")


if __name__ == '__main__':
    main()
