"""
Cellpose Fine-tuning: via_project.json maskelerini kullanarak
Cellpose'u bright-field MSC hucreleri icin fine-tune eder.
"""
import os, sys, json
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon
from cellpose import models, train
import torch

ROOT = r"c:\Users\emir_\Documents\GitHub\Yeni dizin\dataset_senescence"
TRAIN_DIR = os.path.join(ROOT, 'data', 'raw', 'train')
TEST_DIR = os.path.join(ROOT, 'data', 'raw', 'test')
JSON_TRAIN = os.path.join(TRAIN_DIR, 'via_project.json')
JSON_TEST = os.path.join(TEST_DIR, 'via_project.json')
MODEL_SAVE = os.path.join(ROOT, 'checkpoints', 'cellpose')
os.makedirs(MODEL_SAVE, exist_ok=True)

def via_to_cellpose_masks(json_path, img_dir):
    """Convert VIA polygons to Cellpose instance mask format."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    images = []
    masks = []
    
    for key, entry in data.items():
        filename = entry['filename']
        img_path = os.path.join(img_dir, filename)
        if not os.path.exists(img_path):
            continue
        
        img = np.array(Image.open(img_path).convert('RGB'))
        H, W = img.shape[:2]
        
        # Create instance mask (each cell gets unique ID)
        instance_mask = np.zeros((H, W), dtype=np.int32)
        cell_id = 0
        
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
            
            cell_id += 1
            rr, cc = sk_polygon(py, px, shape=(H, W))
            instance_mask[rr, cc] = cell_id
        
        if cell_id > 0:
            images.append(img)
            masks.append(instance_mask)
    
    return images, masks

# Convert ALL data for maximum training (train + val + test = ~100 images)
print("Converting VIA masks to Cellpose format...")
all_images, all_masks = via_to_cellpose_masks(JSON_TRAIN, TRAIN_DIR)

# Add val
VAL_DIR = os.path.join(ROOT, 'data', 'raw', 'val')
JSON_VAL = os.path.join(VAL_DIR, 'via_project.json')
if os.path.exists(JSON_VAL):
    val_imgs, val_msks = via_to_cellpose_masks(JSON_VAL, VAL_DIR)
    all_images.extend(val_imgs)
    all_masks.extend(val_msks)

# Add test
test_imgs, test_msks = via_to_cellpose_masks(JSON_TEST, TEST_DIR)
all_images.extend(test_imgs)
all_masks.extend(test_msks)

# Use last 10 as sanity-check test, rest as train
train_images = all_images[:-10]
train_masks = all_masks[:-10]
test_images = all_images[-10:]
test_masks = all_masks[-10:]
print(f"Train: {len(train_images)} images, Test: {len(test_images)} images (sanity check)")

# Fine-tune Cellpose
print("\nFine-tuning Cellpose on bright-field MSC data...")
model = models.CellposeModel(gpu=True, model_type='cyto2')

# Train using cellpose v3 API
new_model_path = train.train_seg(
    model.net,
    train_data=train_images, 
    train_labels=train_masks,
    test_data=test_images,
    test_labels=test_masks,
    channels=[0, 0],
    save_path=MODEL_SAVE,
    n_epochs=300,
    learning_rate=0.01,
    weight_decay=1e-5,
    min_train_masks=1,
    model_name='cellpose_msc_brightfield',
)

print(f"\nFine-tuned model saved to: {new_model_path}")

# Extract path from tuple (train_seg returns (path, train_losses, test_losses))
if isinstance(new_model_path, tuple):
    model_path_str = str(new_model_path[0])
else:
    model_path_str = str(new_model_path)

# Quick test with fine-tuned model
print("\nTesting fine-tuned model on test set...")
ft_model = models.CellposeModel(gpu=True, pretrained_model=model_path_str)
masks_pred = ft_model.eval(test_images, channels=[0, 0], diameter=None)[0]

# Count matches like before
from skimage.draw import polygon as sk_polygon2
total_gt = sum(m.max() for m in test_masks)
total_pred = sum(m.max() for m in masks_pred)
total_matched = 0

for gt_mask, pred_mask in zip(test_masks, masks_pred):
    num_gt = gt_mask.max()
    num_pred = pred_mask.max()
    matched = 0
    for gt_id in range(1, num_gt + 1):
        gt_m = (gt_mask == gt_id)
        best_iou = 0
        for pred_id in range(1, num_pred + 1):
            pred_m = (pred_mask == pred_id)
            inter = np.logical_and(gt_m, pred_m).sum()
            union = np.logical_or(gt_m, pred_m).sum()
            iou = inter / max(union, 1)
            best_iou = max(best_iou, iou)
        if best_iou > 0.3:
            matched += 1
    total_matched += matched

print(f"\nFINE-TUNED RESULTS:")
print(f"  GT cells: {total_gt}, Predicted: {total_pred}, Matched: {total_matched}")
print(f"  Recall: {total_matched/max(total_gt,1)*100:.1f}%")
print(f"  Precision: {total_matched/max(total_pred,1)*100:.1f}%")
