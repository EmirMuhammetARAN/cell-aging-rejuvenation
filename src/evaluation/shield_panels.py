# -*- coding: utf-8 -*-
"""
Generate 6 Publication-Grade Panels for the 3 Biological Validation Shields
---------------------------------------------------------------------------
1. Shield 1 - Morphometry & Area (Exemplars / Best Cases)
2. Shield 1 - Morphometry & Area (Failure Modes / Worst Cases)
3. Shield 2 - Cytoplasmic Texture & Entropy (Exemplars / Best Cases)
4. Shield 2 - Cytoplasmic Texture & Entropy (Failure Modes / Worst Cases)
5. Shield 3 - Explainable AI & Hierarchical LayerCAM (Exemplars / Best Cases)
6. Shield 3 - Explainable AI & Hierarchical LayerCAM (Failure Modes / Worst Cases)
"""

import os, sys, time, shutil, csv
os.environ['PYTHONIOENCODING'] = 'utf-8'
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
import torch
import torch.nn.functional as F
from torchvision import transforms

base_dir = r"D:\GitHub\cell-aging-rejuvenation"
sys.path.insert(0, base_dir)

from models.classifier.classifier import Classifier
from src.evaluation.shield3_layercam import HierarchicalLayerCAM, compute_alignment_metrics, create_composite_overlay

results_dir = os.path.join(base_dir, 'results', 'morphological_validation')
artifact_dir = r"C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7"
scratch_dir = os.path.join(base_dir, 'scratch')

# Image paths
y_in_dir = os.path.join(base_dir, "data", "processed_v4", "test", "young")
ag_out_dir = os.path.join(base_dir, "results", "generated", "seed_sweep", "seed_2026", "aging")
cg_ag_dir = os.path.join(base_dir, "results", "generated", "cyclegan_v2_sweep", "epoch_160", "aging")

s_in_dir = os.path.join(base_dir, "data", "processed_v4", "test", "senescent")
re_out_dir = os.path.join(base_dir, "results", "generated", "seed_sweep", "seed_2026", "rejuv")
cg_re_dir = os.path.join(base_dir, "results", "generated", "cyclegan_v2_sweep", "epoch_160", "rejuvenation")

# Load Ground Truth Masks
gt_aging = np.load(os.path.join(scratch_dir, 'mrcnn_masks_aging.npy'), allow_pickle=True).item()
gt_reju = np.load(os.path.join(scratch_dir, 'mrcnn_masks_rejuv.npy'), allow_pickle=True).item()

# Load Morphometrics Data
with open(os.path.join(results_dir, "full_test_morphometrics.csv"), 'r') as f:
    morph_data = { (r['Task'], r['Filename']): r for r in csv.DictReader(f) }

# Load Texture Data
with open(os.path.join(results_dir, "full_test_texture_metrics.csv"), 'r') as f:
    tex_data = { (r['Task'], r['Filename']): r for r in csv.DictReader(f) }

# Load LayerCAM Alignment Data
with open(os.path.join(results_dir, "full_test_layercam_alignment.csv"), 'r') as f:
    cam_data = { (r['Task'], r['Filename']): r for r in csv.DictReader(f) }

# Classifier & LayerCAM
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = os.path.join(base_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
classifier = Classifier(output_size=2)
classifier.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
classifier.to(device)
classifier.eval()
layercam = HierarchicalLayerCAM(classifier)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def draw_contour(img, mask, color=(0, 255, 0), thickness=2):
    vis = img.copy()
    if np.sum(mask) > 0:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, thickness)
    return vis


# ==============================================================================
# 1. SHIELD 1: MORPHOMETRIC PANELS (BEST & WORST)
# ==============================================================================
def generate_shield1_panel(samples, title, out_name, is_best=True):
    fig = plt.figure(figsize=(20, 16.5), dpi=300)
    gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.08, hspace=0.20)
    
    col_titles = [
        "Input Image\n(Control Baseline)",
        "CycleGAN Epoch 160\n(Surface Translation)",
        "LDM v12 (Diffusion)\n(Surface Translation)",
        "Physical Deformation\n(|LDM - Input| Heatmap)",
        "Morphometric Validation\n(Expansion / Contraction)"
    ]
    
    for row_idx, (task, fname) in enumerate(samples):
        in_dir = y_in_dir if task == 'Aging' else s_in_dir
        cg_dir = cg_ag_dir if task == 'Aging' else cg_re_dir
        ldm_dir = ag_out_dir if task == 'Aging' else re_out_dir
        mask_dict = gt_aging if task == 'Aging' else gt_reju
        
        img_in = cv2.imread(os.path.join(in_dir, fname))[:, :, ::-1]
        img_cg = cv2.imread(os.path.join(cg_dir, fname))[:, :, ::-1]
        img_ldm = cv2.imread(os.path.join(ldm_dir, fname))[:, :, ::-1]
        
        m_ldm = mask_dict.get(fname, np.zeros(img_in.shape[:2], dtype=bool))
        
        # Morphometrics
        rec = morph_data.get((task, fname), None)
        in_area = int(float(rec['Input_Area'])) if rec else 0
        cg_area = int(float(rec['CycleGAN_Area'])) if rec else 0
        cg_delta = float(rec['CycleGAN_Delta_Area']) if rec else 0.0
        ldm_area = int(float(rec['LDM_Area'])) if rec else int(np.sum(m_ldm))
        ldm_delta = float(rec['LDM_Delta_Area']) if rec else 0.0
        ldm_cdi = float(rec['LDM_CDI']) if rec else 0.0
        
        # Difference map
        diff = np.mean(np.abs(img_ldm.astype(float) - img_in.astype(float)), axis=-1)
        p99 = np.percentile(diff, 99.5)
        diff_clipped = np.clip(diff, 0, p99)
        diff_norm = (diff_clipped - diff_clipped.min()) / (diff_clipped.max() - diff_clipped.min() + 1e-8)
        
        # 1. Input
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(img_in); ax0.set_xticks([]); ax0.set_yticks([])
        if row_idx == 0: ax0.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=10)
        ax0.set_ylabel(f"{task}\n{fname[:22]}", fontsize=10, fontweight='bold')
        ax0.text(0.04, 0.06, f"Area: {in_area:,} px", transform=ax0.transAxes, color='yellow',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 2. CycleGAN
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(img_cg); ax1.set_xticks([]); ax1.set_yticks([])
        if row_idx == 0: ax1.set_title(col_titles[1], fontsize=11, fontweight='bold', pad=10)
        ax1.text(0.04, 0.06, f"Area: {cg_area:,} px\nΔ: {cg_delta:+.1f}%", transform=ax1.transAxes, color='white',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 3. LDM Output with contour
        ax2 = fig.add_subplot(gs[row_idx, 2])
        vis_ldm = draw_contour(img_ldm, m_ldm, color=(0, 255, 0) if is_best else (255, 0, 0), thickness=2)
        ax2.imshow(vis_ldm); ax2.set_xticks([]); ax2.set_yticks([])
        if row_idx == 0: ax2.set_title(col_titles[2], fontsize=11, fontweight='bold', pad=10)
        d_color = '#51cf66' if (task == 'Aging' and ldm_delta > 0) or (task == 'Rejuvenation' and ldm_delta < 0) else '#ff6b6b'
        ax2.text(0.04, 0.06, f"Area: {ldm_area:,} px\nΔ: {ldm_delta:+.1f}%", transform=ax2.transAxes, color=d_color,
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 4. Difference Map
        ax3 = fig.add_subplot(gs[row_idx, 3])
        im3 = ax3.imshow(diff_norm, cmap='inferno')
        ax3.set_xticks([]); ax3.set_yticks([])
        if row_idx == 0: ax3.set_title(col_titles[3], fontsize=11, fontweight='bold', pad=10)
        ax3.text(0.04, 0.06, f"CDI: {ldm_cdi:.2f}x", transform=ax3.transAxes, color='white',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 5. Metric Summary Card
        ax4 = fig.add_subplot(gs[row_idx, 4])
        ax4.set_facecolor('#1e1e1e' if is_best else '#2b1a1a')
        ax4.set_xticks([]); ax4.set_yticks([])
        if row_idx == 0: ax4.set_title(col_titles[4], fontsize=11, fontweight='bold', pad=10)
        
        status_txt = "PHYSIOLOGICAL SUCCESS" if is_best else "DEFORMATION FAILURE / OUTLIER"
        status_color = '#4ade80' if is_best else '#f87171'
        
        summary_lines = [
            f"Direction: {task}",
            f"Input Area: {in_area:,} px",
            f"LDM Area:   {ldm_area:,} px",
            f"Area Delta: {ldm_delta:+.1f}%",
            f"Target CDI: {ldm_cdi:.2f}x",
            f"Status: {status_txt}"
        ]
        y_pos = 0.85
        for s_line in summary_lines:
            c = status_color if "Status" in s_line else 'white'
            fw = 'bold' if ("Status" in s_line or "Area Delta" in s_line) else 'normal'
            ax4.text(0.08, y_pos, s_line, color=c, fontsize=10, fontweight=fw, transform=ax4.transAxes)
            y_pos -= 0.14
            
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    out_path = os.path.join(results_dir, out_name)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    shutil.copy2(out_path, os.path.join(artifact_dir, out_name))
    print(f"[OK] Saved Shield 1 Panel: {out_name}")


# ==============================================================================
# 2. SHIELD 2: TEXTURE & SHANNON ENTROPY PANELS (BEST & WORST)
# ==============================================================================
def generate_shield2_panel(samples, title, out_name, is_best=True):
    fig = plt.figure(figsize=(20, 16.5), dpi=300)
    gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.08, hspace=0.20)
    
    col_titles = [
        "Input RGB Cell\n(Cytoplasmic ROI)",
        "Input Shannon Entropy Map\n(Local Granularity H)",
        "LDM Output RGB Cell\n(Synthesized Cytoplasm)",
        "LDM Shannon Entropy Map\n(Local Granularity H)",
        "Entropy Profile & GLCM\n(Vacuolization / Recovery)"
    ]
    
    for row_idx, (task, fname) in enumerate(samples):
        in_dir = y_in_dir if task == 'Aging' else s_in_dir
        ldm_dir = ag_out_dir if task == 'Aging' else re_out_dir
        
        img_in = cv2.imread(os.path.join(in_dir, fname))[:, :, ::-1]
        img_ldm = cv2.imread(os.path.join(ldm_dir, fname))[:, :, ::-1]
        
        gray_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)
        gray_ldm = cv2.cvtColor(img_ldm, cv2.COLOR_RGB2GRAY)
        
        # Local Shannon entropy maps
        ent_map_in = rank_entropy(gray_in, disk(5))
        ent_map_ldm = rank_entropy(gray_ldm, disk(5))
        
        # Global metrics from CSV
        rec = tex_data.get((task, fname), None)
        h_in = float(rec['In_Entropy']) if rec else 0.0
        h_ldm = float(rec['LDM_Entropy']) if rec else 0.0
        delta_h = h_ldm - h_in
        contrast_in = float(rec['In_Contrast']) if rec else 0.0
        contrast_ldm = float(rec['LDM_Contrast']) if rec else 0.0
        
        # 1. Input RGB
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(img_in); ax0.set_xticks([]); ax0.set_yticks([])
        if row_idx == 0: ax0.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=10)
        ax0.set_ylabel(f"{task}\n{fname[:22]}", fontsize=10, fontweight='bold')
        ax0.text(0.04, 0.06, f"Global H: {h_in:.3f} bits", transform=ax0.transAxes, color='yellow',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 2. Input Entropy Heatmap
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(ent_map_in, cmap='magma'); ax1.set_xticks([]); ax1.set_yticks([])
        if row_idx == 0: ax1.set_title(col_titles[1], fontsize=11, fontweight='bold', pad=10)
        
        # 3. LDM RGB
        ax2 = fig.add_subplot(gs[row_idx, 2])
        ax2.imshow(img_ldm); ax2.set_xticks([]); ax2.set_yticks([])
        if row_idx == 0: ax2.set_title(col_titles[2], fontsize=11, fontweight='bold', pad=10)
        ax2.text(0.04, 0.06, f"Global H: {h_ldm:.3f} bits", transform=ax2.transAxes, color='white',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 4. LDM Entropy Heatmap
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.imshow(ent_map_ldm, cmap='magma'); ax3.set_xticks([]); ax3.set_yticks([])
        if row_idx == 0: ax3.set_title(col_titles[3], fontsize=11, fontweight='bold', pad=10)
        
        # 5. Metric Summary Card
        ax4 = fig.add_subplot(gs[row_idx, 4])
        ax4.set_facecolor('#1e1e1e' if is_best else '#2b1a1a')
        ax4.set_xticks([]); ax4.set_yticks([])
        if row_idx == 0: ax4.set_title(col_titles[4], fontsize=11, fontweight='bold', pad=10)
        
        status_txt = "TEXTURE CONGRUENCE" if is_best else "ANOMALOUS TEXTURE / STAGNANT"
        status_color = '#4ade80' if is_best else '#f87171'
        
        summary_lines = [
            f"Phenotype: {task}",
            f"Input Entropy:  {h_in:.3f} bits",
            f"LDM Entropy:    {h_ldm:.3f} bits",
            f"Delta Entropy:  {delta_h:+.3f} bits",
            f"GLCM Contrast:  {contrast_in:.2f} -> {contrast_ldm:.2f}",
            f"Status: {status_txt}"
        ]
        y_pos = 0.85
        for s_line in summary_lines:
            c = status_color if "Status" in s_line else 'white'
            fw = 'bold' if ("Status" in s_line or "Delta Entropy" in s_line) else 'normal'
            ax4.text(0.08, y_pos, s_line, color=c, fontsize=10, fontweight=fw, transform=ax4.transAxes)
            y_pos -= 0.14
            
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    out_path = os.path.join(results_dir, out_name)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    shutil.copy2(out_path, os.path.join(artifact_dir, out_name))
    print(f"[OK] Saved Shield 2 Panel: {out_name}")


# ==============================================================================
# 3. SHIELD 3: HIERARCHICAL LAYERCAM PANELS (BEST & WORST)
# ==============================================================================
def generate_shield3_panel(samples, title, out_name, is_best=True):
    fig = plt.figure(figsize=(20, 16.5), dpi=300)
    gs = gridspec.GridSpec(4, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.08, hspace=0.20)
    
    col_titles = [
        "Input Cell Image\n(Source Phenotype)",
        "LDM Translated Output\n(w/ ResNet-18 Decision)",
        "Ground-Truth Mask R-CNN\n(Single-Cell Contour)",
        "Hierarchical LayerCAM\n(Layer 3+4 Target Heatmap)",
        "Cross-Modal Alignment\n(Green: GT Mask, Red: CAM, Yellow: Overlap)"
    ]
    
    for row_idx, (task, fname) in enumerate(samples):
        in_dir = y_in_dir if task == 'Aging' else s_in_dir
        ldm_dir = ag_out_dir if task == 'Aging' else re_out_dir
        mask_dict = gt_aging if task == 'Aging' else gt_reju
        target_class = 0 if task == 'Aging' else 1
        
        img_in = cv2.imread(os.path.join(in_dir, fname))[:, :, ::-1]
        img_ldm = cv2.imread(os.path.join(ldm_dir, fname))[:, :, ::-1]
        pil_ldm = Image.open(os.path.join(ldm_dir, fname)).convert('RGB')
        
        cell_m = mask_dict.get(fname, np.zeros(img_in.shape[:2], dtype=bool))
        
        # LayerCAM execution on GPU
        tensor_ldm = transform(pil_ldm).unsqueeze(0).to(device)
        cam, pred_class, conf = layercam.generate(tensor_ldm, target_class=target_class)
        
        # Real metrics from CSV
        rec = cam_data.get((task, fname), None)
        mask_iou = float(rec['Mask_IoU']) if rec else 0.0
        bbox_iou = float(rec['BBox_IoU']) if rec else 0.0
        energy = float(rec['In_Cell_Energy']) if rec else 0.0
        pred_label = rec['Predicted_Class'] if rec else ('Senescent' if pred_class == 0 else 'Young')
        
        # Composite Overlay (Green: GT Mask, Red: CAM Attention, Yellow: Overlap)
        cam_resized = cv2.resize(cam, (512, 512))
        overlay = np.zeros((512, 512, 3), dtype=np.float32)
        overlay[:, :, 0] = cam_resized * 255.0                  # Red = Attention
        overlay[:, :, 1] = cell_m.astype(np.float32) * 255.0    # Green = True Mask
        overlay[:, :, 2] = 0
        
        gray_base = cv2.cvtColor(img_ldm, cv2.COLOR_RGB2GRAY)
        gray_3ch = np.stack([gray_base]*3, axis=-1).astype(np.float32)
        blend = np.clip(0.40 * gray_3ch + 0.60 * overlay, 0, 255).astype(np.uint8)
        
        # 1. Input
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0.imshow(img_in); ax0.set_xticks([]); ax0.set_yticks([])
        if row_idx == 0: ax0.set_title(col_titles[0], fontsize=11, fontweight='bold', pad=10)
        ax0.set_ylabel(f"{task}\n{fname[:22]}", fontsize=10, fontweight='bold')
        
        # 2. LDM Translated Output + Badge
        ax1 = fig.add_subplot(gs[row_idx, 1])
        ax1.imshow(img_ldm); ax1.set_xticks([]); ax1.set_yticks([])
        if row_idx == 0: ax1.set_title(col_titles[1], fontsize=11, fontweight='bold', pad=10)
        badge_color = 'green' if (task == 'Aging' and pred_class == 0) or (task == 'Rejuvenation' and pred_class == 1) else 'red'
        ax1.text(0.04, 0.06, f"Pred: {pred_label} ({conf*100:.1f}%)", transform=ax1.transAxes, color='white',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_color, alpha=0.8))
        
        # 3. Ground Truth Mask R-CNN Contour
        ax2 = fig.add_subplot(gs[row_idx, 2])
        vis_gt = draw_contour(img_ldm, cell_m, color=(0, 255, 0), thickness=2)
        ax2.imshow(vis_gt); ax2.set_xticks([]); ax2.set_yticks([])
        if row_idx == 0: ax2.set_title(col_titles[2], fontsize=11, fontweight='bold', pad=10)
        ax2.text(0.04, 0.06, f"GT Area: {int(np.sum(cell_m)):,} px", transform=ax2.transAxes, color='yellow',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 4. LayerCAM Heatmap
        ax3 = fig.add_subplot(gs[row_idx, 3])
        ax3.imshow(cam_resized, cmap='jet'); ax3.set_xticks([]); ax3.set_yticks([])
        if row_idx == 0: ax3.set_title(col_titles[3], fontsize=11, fontweight='bold', pad=10)
        ax3.text(0.04, 0.06, f"In-Cell Energy: {energy:.1f}%", transform=ax3.transAxes, color='white',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        # 5. Composite Alignment
        ax4 = fig.add_subplot(gs[row_idx, 4])
        ax4.imshow(blend); ax4.set_xticks([]); ax4.set_yticks([])
        if row_idx == 0: ax4.set_title(col_titles[4], fontsize=11, fontweight='bold', pad=10)
        
        iou_badge = f"Mask IoU: {mask_iou:.1f}%\nBBox IoU: {bbox_iou:.1f}%"
        ax4.text(0.04, 0.06, iou_badge, transform=ax4.transAxes, color='white',
                 fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))
        
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    out_path = os.path.join(results_dir, out_name)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    shutil.copy2(out_path, os.path.join(artifact_dir, out_name))
    print(f"[OK] Saved Shield 3 Panel: {out_name}")


def main():
    print("=" * 80)
    print("GENERATING 6 PUBLICATION PANELS FOR 3 VALIDATION SHIELDS")
    print("=" * 80)
    
    # --------------------------------------------------------------------------
    # 1. SHIELD 1: Morphometry & Area
    # --------------------------------------------------------------------------
    s1_best = [
        ('Aging', 'senescent_MSCs_10005_13.jpg'),
        ('Aging', 'young_MSCs_10044_625.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10102_289.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10168_343.jpg')
    ]
    s1_worst = [
        ('Aging', 'young_MSCs_10040_512.jpg'),
        ('Aging', 'young_MSCs_10047_635.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10005_10.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10031_63.jpg')
    ]
    generate_shield1_panel(s1_best, "Shield 1: Single-Cell Morphometry & Area Dynamics (Exemplar Transitions)",
                           "figure_shield1_morphometry_best.png", is_best=True)
    generate_shield1_panel(s1_worst, "Shield 1: Morphological Failure Modes & Outlier Boundary Cases",
                           "figure_shield1_morphometry_worst.png", is_best=False)
                           
    # --------------------------------------------------------------------------
    # 2. SHIELD 2: Texture & Entropy
    # --------------------------------------------------------------------------
    s2_best = [
        ('Aging', 'young_MSCs_10041_533.jpg'),
        ('Aging', 'young_MSCs_10016_424.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10058_142.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10090_242.jpg')
    ]
    s2_worst = [
        ('Aging', 'senescent_MSCs_10066_183.jpg'),
        ('Aging', 'senescent_MSCs_10040_77.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10005_10.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10031_63.jpg')
    ]
    generate_shield2_panel(s2_best, "Shield 2: Cytoplasmic Texture & Shannon Entropy Dynamics (Exemplar Reversals)",
                           "figure_shield2_texture_best.png", is_best=True)
    generate_shield2_panel(s2_worst, "Shield 2: Intracellular Texture Failure Modes & Stagnant Granularity",
                           "figure_shield2_texture_worst.png", is_best=False)
                           
    # --------------------------------------------------------------------------
    # 3. SHIELD 3: Hierarchical LayerCAM
    # --------------------------------------------------------------------------
    s3_best = [
        ('Aging', 'senescent_MSCs_10005_11.jpg'),
        ('Aging', 'senescent_MSCs_10005_16.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10063_154.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10097_253.jpg')
    ]
    s3_worst = [
        ('Aging', 'young_MSCs_10022_467.jpg'),
        ('Aging', 'young_MSCs_10077_666.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10005_10.jpg'),
        ('Rejuvenation', 'senescent_MSCs_10097_259.jpg')
    ]
    generate_shield3_panel(s3_best, "Shield 3: Explainable AI & Hierarchical LayerCAM Alignment (Exemplar Visualizations)",
                           "figure_shield3_layercam_best.png", is_best=True)
    generate_shield3_panel(s3_worst, "Shield 3: Spatial Misalignment Modes & Low-Saliency Edge Cases",
                           "figure_shield3_layercam_worst.png", is_best=False)
                           
    print("\n[FINISHED] All 6 publication panels generated and synced to artifact directory!")

if __name__ == '__main__':
    main()

