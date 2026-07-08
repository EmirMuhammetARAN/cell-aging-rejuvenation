"""
Mask R-CNN Precision/Recall evaluation script.
Runs inference on labeled data and compares with ground truth via_project.json.
"""
import os
import sys
import json
import cv2
import numpy as np
import skimage.io
import time

import mrcnn.model as modellib
from mrcnn.config import Config


class CustomConfig(Config):
    NAME = "object"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.7

class InferenceConfig(CustomConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.7


def mask_from_regions(regions, shape, target_class=None):
    """Render all polygon regions into a binary mask."""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for r in regions:
        sa = r["shape_attributes"]
        cell = r.get("region_attributes", {}).get("cell", "")
        if target_class and cell != target_class:
            continue
        if sa.get("name") == "polygon":
            pts_x = sa["all_points_x"]
            pts_y = sa["all_points_y"]
            pts = np.array(list(zip(pts_x, pts_y)), np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def extract_polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) > 5:
            contour = contour.squeeze(1)
            poly = {
                "name": "polygon",
                "all_points_x": [int(x) for x in contour[:, 0]],
                "all_points_y": [int(y) for y in contour[:, 1]]
            }
            polygons.append(poly)
    return polygons


def count_cells_by_class(regions):
    counts = {}
    for r in regions:
        cell = r.get("region_attributes", {}).get("cell", "unknown")
        counts[cell] = counts.get(cell, 0) + 1
    return counts


def compute_iou_mask(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0


def compute_precision_recall_mask(gt_mask, pred_mask):
    tp = np.logical_and(gt_mask, pred_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask.astype(bool)).sum()
    fn = np.logical_and(gt_mask, ~pred_mask.astype(bool)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, int(tp), int(fp), int(fn)


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_path = os.path.join(root_dir, "data", "raw")
    weights_path = os.path.join(root_dir, "fatma hoca", "mask_rcnn_object_0800.h5")
    
    config = InferenceConfig()
    
    print("Loading model...")
    model = modellib.MaskRCNN(mode="inference", model_dir="logs", config=config)
    model.keras_model.load_weights(weights_path, by_name=True)
    print("Model loaded!\n")
    
    class_names = ['BG', 'young', 'senescent']
    subsets = ["train", "val", "test"]
    
    # Global accumulators
    all_gt_cells = 0
    all_pred_cells = 0
    all_tp_pixels = 0
    all_fp_pixels = 0
    all_fn_pixels = 0
    
    per_class_stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in ["young", "senescent"]}
    
    total_images = 0
    total_time = 0
    
    for subset in subsets:
        subset_path = os.path.join(raw_path, subset)
        gt_json_path = os.path.join(subset_path, "via_project.json")
        
        if not os.path.exists(gt_json_path):
            print(f"[WARN] {gt_json_path} not found, skipping {subset}")
            continue
        
        # Load ground truth
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        # Build GT lookup by filename
        gt_by_file = {}
        for key, val in gt_data.items():
            fname = val["filename"]
            gt_by_file[fname] = val.get("regions", [])
        
        image_files = [f for f in os.listdir(subset_path) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        print(f"=== {subset.upper()} === ({len(image_files)} images)")
        
        subset_gt_cells = 0
        subset_pred_cells = 0
        
        for idx, filename in enumerate(image_files):
            image_path = os.path.join(subset_path, filename)
            try:
                image = skimage.io.imread(image_path)
                if image.ndim == 2:
                    image = skimage.color.gray2rgb(image)
                elif image.shape[-1] == 4:
                    image = image[..., :3]
            except Exception as e:
                print(f"  Failed to read {filename}: {e}")
                continue
            
            h, w = image.shape[:2]
            
            t0 = time.time()
            results = model.detect([image], verbose=0)
            dt = time.time() - t0
            total_time += dt
            total_images += 1
            
            r = results[0]
            
            # Build pred regions
            pred_regions = []
            for i in range(r['rois'].shape[0]):
                class_id = r['class_ids'][i]
                score = r['scores'][i]
                mask = r['masks'][:, :, i]
                label_str = class_names[class_id]
                polygons = extract_polygons(mask)
                for poly in polygons:
                    pred_regions.append({
                        "shape_attributes": poly,
                        "region_attributes": {"cell": label_str, "score": float(score)}
                    })
            
            # Get GT regions
            gt_regions = gt_by_file.get(filename, [])
            
            subset_gt_cells += len(gt_regions)
            subset_pred_cells += len(pred_regions)
            
            # Per-class pixel comparison
            for cls in ["young", "senescent"]:
                gt_m = mask_from_regions(gt_regions, (h, w), target_class=cls)
                pred_m = mask_from_regions(pred_regions, (h, w), target_class=cls)
                
                tp = np.logical_and(gt_m, pred_m).sum()
                fp = np.logical_and(pred_m, ~gt_m.astype(bool)).sum()
                fn = np.logical_and(gt_m, ~pred_m.astype(bool)).sum()
                
                per_class_stats[cls]["tp"] += int(tp)
                per_class_stats[cls]["fp"] += int(fp)
                per_class_stats[cls]["fn"] += int(fn)
                
                all_tp_pixels += int(tp)
                all_fp_pixels += int(fp)
                all_fn_pixels += int(fn)
            
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(image_files)}] {filename} | "
                      f"GT: {len(gt_regions)} cells, Pred: {len(pred_regions)} cells | "
                      f"{dt:.2f}s")
        
        all_gt_cells += subset_gt_cells
        all_pred_cells += subset_pred_cells
        
        print(f"  Subset total: GT={subset_gt_cells} cells, Pred={subset_pred_cells} cells\n")
    
    # Final report
    print("=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total time: {total_time:.1f}s ({total_time/total_images:.2f}s/image)")
    print(f"Total GT cells: {all_gt_cells}")
    print(f"Total Pred cells: {all_pred_cells}")
    print()
    
    for cls in ["young", "senescent"]:
        s = per_class_stats[cls]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"[{cls.upper()}]")
        print(f"  Pixel Precision: {prec:.4f}")
        print(f"  Pixel Recall:    {rec:.4f}")
        print(f"  Pixel F1:        {f1:.4f}")
        print()
    
    # Overall
    prec_all = all_tp_pixels / (all_tp_pixels + all_fp_pixels) if (all_tp_pixels + all_fp_pixels) > 0 else 0
    rec_all = all_tp_pixels / (all_tp_pixels + all_fn_pixels) if (all_tp_pixels + all_fn_pixels) > 0 else 0
    f1_all = 2 * prec_all * rec_all / (prec_all + rec_all) if (prec_all + rec_all) > 0 else 0
    iou_all = all_tp_pixels / (all_tp_pixels + all_fp_pixels + all_fn_pixels) if (all_tp_pixels + all_fp_pixels + all_fn_pixels) > 0 else 0
    
    print(f"[OVERALL]")
    print(f"  Pixel Precision: {prec_all:.4f}")
    print(f"  Pixel Recall:    {rec_all:.4f}")
    print(f"  Pixel F1:        {f1_all:.4f}")
    print(f"  Pixel IoU:       {iou_all:.4f}")


if __name__ == "__main__":
    main()
