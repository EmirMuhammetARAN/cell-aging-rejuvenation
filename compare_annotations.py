import json
import cv2
import numpy as np
import os

GT_JSON = r"data\raw\val\via_project.json"
PRED_JSON = r"data\unlabeled_test\via_project.json"
IMG_NAME = "young_MSCs_10070.jpg"

def get_regions(json_path, target_img):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for key, val in data.items():
        if val["filename"] == target_img:
            return val["regions"]
    return []

def draw_polygons(img_shape, regions):
    mask = np.zeros(img_shape, dtype=np.uint8)
    for r in regions:
        shape = r["shape_attributes"]
        if shape["name"] == "polygon":
            pts = []
            for x, y in zip(shape["all_points_x"], shape["all_points_y"]):
                pts.append([x, y])
            pts = np.array(pts, np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask

gt_regions = get_regions(GT_JSON, IMG_NAME)
pred_regions = get_regions(PRED_JSON, IMG_NAME)

print(f"Ground Truth cell count: {len(gt_regions)}")
print(f"Predicted cell count: {len(pred_regions)}")

# Count cells by class
gt_young = sum(1 for r in gt_regions if r["region_attributes"].get("cell") == "young")
gt_senes = sum(1 for r in gt_regions if r["region_attributes"].get("cell") == "senescent")

pred_young = sum(1 for r in pred_regions if r["region_attributes"].get("cell") == "young")
pred_senes = sum(1 for r in pred_regions if r["region_attributes"].get("cell") == "senescent")

print(f"GT: {gt_young} young, {gt_senes} senescent")
print(f"Pred: {pred_young} young, {pred_senes} senescent")

# Let's compute IoU roughly by rendering masks (assuming 2592x1944)
shape = (1944, 2592)
gt_mask = draw_polygons(shape, gt_regions)
pred_mask = draw_polygons(shape, pred_regions)

intersection = np.logical_and(gt_mask, pred_mask).sum()
union = np.logical_or(gt_mask, pred_mask).sum()
iou = intersection / union if union > 0 else 0

print(f"Overall Mask IoU: {iou:.4f}")
