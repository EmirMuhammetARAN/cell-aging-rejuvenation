"""
Cellpose Segmentasyon Test: via_project.json ground truth ile karsilastirma.
Cellpose pretrained modeli ile hucreleri bulup, 
VIA maskelerindeki hucre sayisi ve konumlariyla IoU hesaplar.
"""
import os, sys, json
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon
from cellpose import models
import torch

ROOT = r"c:\Users\emir_\Documents\GitHub\Yeni dizin\dataset_senescence"
TEST_DIR = os.path.join(ROOT, 'data', 'raw', 'test')
JSON_PATH = os.path.join(TEST_DIR, 'via_project.json')
OUT_DIR = os.path.join(ROOT, 'results', 'cellpose_test')
os.makedirs(OUT_DIR, exist_ok=True)

# Load Cellpose model (cyto2 = best general cell model)
print("Loading Cellpose model...")
cp_model = models.Cellpose(gpu=True, model_type='cyto2')

# Load ground truth
with open(JSON_PATH, 'r') as f:
    via_data = json.load(f)

print(f"Ground truth images: {len(via_data)}")

total_gt_cells = 0
total_cp_cells = 0
total_matched = 0
iou_scores = []

for key, entry in via_data.items():
    filename = entry['filename']
    img_path = os.path.join(TEST_DIR, filename)
    
    if not os.path.exists(img_path):
        continue
    
    img = np.array(Image.open(img_path).convert('RGB'))
    H, W = img.shape[:2]
    
    # --- Ground truth masks from VIA ---
    gt_masks = []
    gt_classes = []
    for region in entry.get('regions', []):
        shape = region.get('shape_attributes', {})
        attrs = region.get('region_attributes', {})
        cell_class = attrs.get('cell', 'unknown')
        
        if cell_class not in ('young', 'senescent'):
            continue
        
        px = shape.get('all_points_x', [])
        py = shape.get('all_points_y', [])
        if len(px) < 3:
            continue
        
        # Create binary mask
        mask = np.zeros((H, W), dtype=bool)
        rr, cc = sk_polygon(py, px, shape=(H, W))
        mask[rr, cc] = True
        gt_masks.append(mask)
        gt_classes.append(cell_class)
    
    if not gt_masks:
        continue
    
    # --- Cellpose segmentation ---
    masks_cp, flows, styles, diams = cp_model.eval(
        img, diameter=None, channels=[0, 0],  # grayscale
        flow_threshold=0.4, cellprob_threshold=0.0
    )
    
    # Count cellpose detections
    num_cp = masks_cp.max()
    num_gt = len(gt_masks)
    
    # Match: for each GT mask, find best overlapping Cellpose mask
    matched = 0
    for gt_mask in gt_masks:
        best_iou = 0
        # Check each cellpose cell
        for cp_id in range(1, num_cp + 1):
            cp_mask = (masks_cp == cp_id)
            intersection = np.logical_and(gt_mask, cp_mask).sum()
            union = np.logical_or(gt_mask, cp_mask).sum()
            iou = intersection / max(union, 1)
            best_iou = max(best_iou, iou)
        
        if best_iou > 0.3:  # IoU > 0.3 = matched
            matched += 1
            iou_scores.append(best_iou)
    
    total_gt_cells += num_gt
    total_cp_cells += num_cp
    total_matched += matched
    
    recall = matched / max(num_gt, 1) * 100
    print(f"  {filename}: GT={num_gt}, Cellpose={num_cp}, Matched={matched}, Recall={recall:.0f}%")

print(f"\n{'='*60}")
print(f"CELLPOSE SEGMENTATION RESULTS")
print(f"{'='*60}")
print(f"Total GT cells:       {total_gt_cells}")
print(f"Total Cellpose cells: {total_cp_cells}")
print(f"Matched (IoU>0.3):    {total_matched}")
print(f"Recall:               {total_matched/max(total_gt_cells,1)*100:.1f}%")
print(f"Precision:            {total_matched/max(total_cp_cells,1)*100:.1f}%")
if iou_scores:
    print(f"Mean IoU:             {np.mean(iou_scores):.3f}")
print(f"{'='*60}")
