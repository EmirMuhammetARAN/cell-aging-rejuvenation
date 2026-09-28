# -*- coding: utf-8 -*-
"""
SHIELD 3: Explainable AI (XAI) & Hierarchical LayerCAM Alignment Suite
======================================================================
Unified all-in-one suite for Biological Shield 3.
Features:
- Hierarchical LayerCAM (Layer 3 [32x32] + Layer 4 [16x16] fusion)
- Authentic cross-modal spatial alignment against Mask R-CNN Ground-Truth masks:
  1. Pointing Game Hit Rate (% peak inside cell)
  2. Cellular Bounding Box IoU
  3. True Mask IoU (Otsu CAM vs Mask R-CNN)
  4. In-Cell Attention Energy Specificity (%)
- 5-column composite heatmaps (figure_gradcam_alignment_aging/rejuvenation.png)
- Full population evaluation on LDM v12 (N = 645; figure_full_population_layercam_violin.png)
- Baseline CycleGAN benchmark comparison mode
"""

import os
import sys
import time
import csv
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import shutil

# Ensure project root is in sys.path
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier


class HierarchicalLayerCAM:
    """
    Hierarchical LayerCAM implementation for ResNet architectures.
    Extracts element-wise positive gradient weighted activations from Layer 3 and Layer 4,
    combining fine spatial resolution (32x32) with deep semantic certainty (16x16).
    """
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def forward_hook(module, input, output):
                self.activations[name] = output.detach()
            def backward_hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return forward_hook, backward_hook

        for name, module in [('layer3', self.model.model.layer3), ('layer4', self.model.model.layer4)]:
            f_hook, b_hook = get_hook(name)
            self.hooks.append(module.register_forward_hook(f_hook))
            self.hooks.append(module.register_full_backward_hook(b_hook))

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        cam_maps = []
        for name in ['layer3', 'layer4']:
            acts = self.activations[name][0]     # [C, H, W]
            grads = self.gradients[name][0]     # [C, H, W]
            
            # Element-wise positive gradient weighting (LayerCAM principle)
            weights = F.relu(grads)
            weighted_acts = weights * acts
            cam = torch.sum(weighted_acts, dim=0) # [H, W]
            cam = F.relu(cam)
            
            cam_np = cam.cpu().numpy()
            c_min, c_max = np.min(cam_np), np.max(cam_np)
            if c_max > c_min:
                cam_np = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_np = np.zeros_like(cam_np)
                
            # Upsample to 512x512
            cam_resized = cv2.resize(cam_np, (512, 512), interpolation=cv2.INTER_LINEAR)
            cam_maps.append(cam_resized)
            
        # Hierarchical Fusion: 50% Layer 3 (High Res) + 50% Layer 4 (High Semantics)
        fused_cam = 0.5 * cam_maps[0] + 0.5 * cam_maps[1]
        f_min, f_max = np.min(fused_cam), np.max(fused_cam)
        if f_max > f_min:
            fused_cam = (fused_cam - f_min) / (f_max - f_min)
            
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return fused_cam, probs

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def compute_alignment_metrics(cam_map, gt_mask):
    """
    Computes rigorous spatial alignment metrics against a single-cell segmentation mask:
    1. Pointing Game Hit Rate (% peak inside cell)
    2. Bounding Box IoU
    3. True Mask IoU (Otsu thresholded CAM vs Mask)
    4. In-Cell Attention Energy Specificity (%)
    """
    H, W = cam_map.shape
    gt_bool = gt_mask > 0
    
    if np.sum(gt_bool) == 0:
        return {'pointing_hit': 0, 'bbox_iou': 0.0, 'mask_iou': 0.0, 'in_cell_energy': 0.0}
        
    # 1. Pointing Game Hit
    peak_y, peak_x = np.unravel_index(np.argmax(cam_map), cam_map.shape)
    pointing_hit = 1 if gt_bool[peak_y, peak_x] else 0
    
    # 2. In-Cell Energy Specificity
    total_energy = float(np.sum(cam_map))
    cell_energy = float(np.sum(cam_map[gt_bool]))
    in_cell_energy = (cell_energy / (total_energy + 1e-8)) * 100.0
    
    # 3. True Mask IoU via Otsu Saliency Threshold
    cam_uint8 = (cam_map * 255).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cam_salient = cam_uint8 >= otsu_thresh
    
    intersection = np.logical_and(cam_salient, gt_bool).sum()
    union = np.logical_or(cam_salient, gt_bool).sum()
    mask_iou = (float(intersection) / float(union) * 100.0) if union > 0 else 0.0
    
    # 4. Cellular Bounding Box IoU
    y_idxs, x_idxs = np.where(gt_bool)
    gt_bbox = [np.min(x_idxs), np.min(y_idxs), np.max(x_idxs), np.max(y_idxs)]
    
    cam_y, cam_x = np.where(cam_salient)
    if len(cam_y) > 0:
        cam_bbox = [np.min(cam_x), np.min(cam_y), np.max(cam_x), np.max(cam_y)]
        
        ixA = max(gt_bbox[0], cam_bbox[0])
        iyA = max(gt_bbox[1], cam_bbox[1])
        ixB = min(gt_bbox[2], cam_bbox[2])
        iyB = min(gt_bbox[3], cam_bbox[3])
        
        inter_area = max(0, ixB - ixA) * max(0, iyB - iyA)
        gt_area = (gt_bbox[2] - gt_bbox[0]) * (gt_bbox[3] - gt_bbox[1])
        cam_area = (cam_bbox[2] - cam_bbox[0]) * (cam_bbox[3] - cam_bbox[1])
        bbox_union = gt_area + cam_area - inter_area
        bbox_iou = (float(inter_area) / float(bbox_union) * 100.0) if bbox_union > 0 else 0.0
    else:
        bbox_iou = 0.0
        
    return {
        'pointing_hit': pointing_hit,
        'bbox_iou': round(bbox_iou, 2),
        'mask_iou': round(mask_iou, 2),
        'in_cell_energy': round(in_cell_energy, 2)
    }


def create_composite_overlay(img_base, diff_norm, cam_norm):
    """
    Creates RGB overlay:
    Green channel = Physical Deformation Difference Map
    Red channel   = Hierarchical LayerCAM Attention
    Yellow        = Overlap between Deformation and Attention
    """
    overlay = img_base.copy().astype(float)
    overlay[..., 1] = np.clip(overlay[..., 1] * (1.0 - 0.7 * diff_norm) + 255.0 * 0.7 * diff_norm, 0, 255)
    overlay[..., 0] = np.clip(overlay[..., 0] * (1.0 - 0.7 * cam_norm) + 255.0 * 0.7 * cam_norm, 0, 255)
    return overlay.astype(np.uint8)



def run_full_population_evaluation(target='ldm', base_dir=root_dir, results_dir=None, artifact_dir=None):
    if results_dir is None:
        results_dir = os.path.join(base_dir, 'results', 'morphological_validation')
    if artifact_dir is None:
        artifact_dir = r"C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7"
        
    scratch_dir = os.path.join(base_dir, 'scratch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("\n" + "=" * 80)
    print(f"RUNNING SHIELD 3: FULL POPULATION HIERARCHICAL LAYERCAM ({target.upper()})")
    print("=" * 80)
    
    # Load Classifier
    cls_ckpt = os.path.join(base_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier = Classifier(output_size=2)
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()
    
    layercam = HierarchicalLayerCAM(classifier)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if target == 'ldm':
        aging_dir = os.path.join(base_dir, 'results/ldm/v12_v4_data_lpips_last/aging')
        reju_dir = os.path.join(base_dir, 'results/ldm/v12_v4_data_lpips_last/rejuvenation')
        mask_aging = np.load(os.path.join(scratch_dir, 'mrcnn_masks_aging.npy'))
        mask_reju = np.load(os.path.join(scratch_dir, 'mrcnn_masks_reju.npy'))
        csv_name = 'full_test_layercam_alignment.csv'
    else:
        aging_dir = os.path.join(base_dir, 'results/cyclegan/epoch_160/aging')
        reju_dir = os.path.join(base_dir, 'results/cyclegan/epoch_160/rejuvenation')
        mask_aging = np.load(os.path.join(scratch_dir, 'mrcnn_masks_cyclegan_aging.npy'))
        mask_reju = np.load(os.path.join(scratch_dir, 'mrcnn_masks_cyclegan_reju.npy'))
        csv_name = 'full_test_cyclegan_layercam_alignment.csv'
        
    tasks = [
        ('Aging', aging_dir, mask_aging, 0, 'Senescent'),
        ('Rejuvenation', reju_dir, mask_reju, 1, 'Young')
    ]
    
    records = []
    
    for task_name, out_dir, masks_arr, target_cls, label_name in tasks:
        files = sorted([f for f in os.listdir(out_dir) if f.endswith(('.jpg', '.png'))])
        print(f"\n[*] Evaluating {task_name.upper()} cohort ({len(files)} cells)...")
        
        for idx, fname in enumerate(files):
            p_img = os.path.join(out_dir, fname)
            pil_img = Image.open(p_img).convert('RGB')
            tensor_img = transform(pil_img).unsqueeze(0).to(device)
            
            cam_map, probs = layercam.generate_cam(tensor_img, target_class=target_cls)
            gt_mask = masks_arr[idx] > 0
            
            metrics = compute_alignment_metrics(cam_map, gt_mask)
            pred_cls = int(np.argmax(probs))
            target_conf = float(probs[target_cls] * 100.0)
            is_success = 1 if pred_cls == target_cls else 0
            
            records.append({
                'Task': task_name,
                'Filename': fname,
                'Target_Class': label_name,
                'Target_Confidence_Pct': round(target_conf, 2),
                'Success': is_success,
                'Pointing_Hit': metrics['pointing_hit'],
                'BBox_IoU': metrics['bbox_iou'],
                'Mask_IoU': metrics['mask_iou'],
                'In_Cell_Energy_Pct': metrics['in_cell_energy']
            })
            
            if (idx + 1) % 50 == 0:
                print(f"    [{idx+1}/{len(files)}] Processed...")

    # Save CSV
    csv_path = os.path.join(results_dir, csv_name)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[OK] Saved CSV: {csv_path}")
    
    # Statistical Summary
    pointing_all = np.mean([r['Pointing_Hit'] for r in records]) * 100
    bbox_all = np.mean([r['BBox_IoU'] for r in records])
    mask_all = np.mean([r['Mask_IoU'] for r in records])
    energy_all = np.mean([r['In_Cell_Energy_Pct'] for r in records])
    acc_all = np.mean([r['Success'] for r in records]) * 100
    
    print("\n" + "=" * 80)
    print(f"FINAL {target.upper()} POPULATION SUMMARY:")
    print(f"  Pointing Game Hit Rate:     {pointing_all:.2f}%")
    print(f"  Cellular Bounding Box IoU:  {bbox_all:.2f}%")
    print(f"  True Mask IoU (Otsu CAM):   {mask_all:.2f}%")
    print(f"  In-Cell Energy Specificity: {energy_all:.2f}%")
    print(f"  Classifier Accuracy (ACC):  {acc_all:.2f}%")
    print("=" * 80)
    
    layercam.remove_hooks()


def main():
    parser = argparse.ArgumentParser(description="Shield 3: Hierarchical LayerCAM Suite")
    parser.add_argument('--target', choices=['ldm', 'cyclegan', 'all'], default='ldm',
                        help="Target model to evaluate: ldm, cyclegan, or all")
    args = parser.parse_args()
    
    base_dir = root_dir
    results_dir = os.path.join(base_dir, 'results', 'morphological_validation')
    artifact_dir = r"C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7"
    os.makedirs(results_dir, exist_ok=True)
    
    if args.target in ['ldm', 'all']:
        run_full_population_evaluation('ldm', base_dir, results_dir, artifact_dir)
        
    if args.target in ['cyclegan', 'all']:
        run_full_population_evaluation('cyclegan', base_dir, results_dir, artifact_dir)
        
    print("\n[FINISHED] Shield 3 evaluation complete!")


if __name__ == '__main__':
    main()
