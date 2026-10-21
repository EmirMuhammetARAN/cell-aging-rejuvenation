"""
Benchmark Cellpose (cyto2) vs. Custom Mask R-CNN on MSC Brightfield Microscopy
==============================================================================
Empirical validation against manual expert annotations (via_project.json)
demonstrating why generalist cell segmentation foundation models (Cellpose)
fail on flat, low-contrast, non-globular brightfield mesenchymal stem cells.

Results on N=185 Ground-Truth Test Cells across 10 full FOVs (2592x1944):
  - Cellpose Recall (IoU >= 0.5): 16.2%  (Misses 83.8% of MSC cells)
  - Cellpose Precision:          23.8%
  - Custom Mask R-CNN Baseline:  Dice 84.4%, IoU 73.0%
"""

import os
import sys
import json
import time
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon
import cellpose.models as cp_models

def run_cellpose_benchmark(num_images=10):
    current_file = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    test_dir = os.path.join(root_dir, 'data', 'raw', 'test')
    json_path = os.path.join(test_dir, 'via_project.json')

    with open(json_path, 'r', encoding='utf-8') as f:
        via_data = json.load(f)

    print(f"[*] Loaded VIA ground truth: {len(via_data)} image entries.", flush=True)

    print("[*] Initializing Cellpose cyto2 model on CPU...", flush=True)
    t0 = time.time()
    model = cp_models.CellposeModel(gpu=False, model_type='cyto2')
    print(f"    Loaded model in {time.time() - t0:.2f}s", flush=True)

    eval_images = list(via_data.items())[:num_images]

    total_gt = 0
    total_cp = 0
    matched_03 = 0
    matched_05 = 0
    all_ious = []

    print("\n" + "=" * 90, flush=True)
    print(f"{'Filename':<30} | {'GT':<4} | {'Cellpose':<8} | {'Match(0.3)':<10} | {'Match(0.5)':<10} | {'Recall':<8} | {'Time':<6}", flush=True)
    print("-" * 90, flush=True)

    for idx, (key, entry) in enumerate(eval_images, 1):
        fn = entry['filename']
        fp = os.path.join(test_dir, fn)
        if not os.path.exists(fp):
            continue

        img_pil = Image.open(fp).convert('RGB')
        w, h = img_pil.size
        img_arr = np.array(img_pil)

        gt_label_map = np.zeros((h, w), dtype=np.int32)
        valid_gt_count = 0
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
            valid_gt_count += 1
            rr, cc = sk_polygon(py, px, shape=(h, w))
            gt_label_map[rr, cc] = valid_gt_count

        if valid_gt_count == 0:
            continue

        t_start = time.time()
        res = model.eval(img_arr, diameter=70.0, channels=[0, 0], flow_threshold=0.4, cellprob_threshold=0.0)
        cp_mask = res[0].astype(np.int32)
        elapsed = time.time() - t_start

        num_cp = int(cp_mask.max())
        num_gt = valid_gt_count

        if num_cp > 0:
            flat_gt = gt_label_map.ravel()
            flat_cp = cp_mask.ravel()
            cross_id = flat_gt * (num_cp + 1) + flat_cp
            overlap = np.bincount(cross_id, minlength=(num_gt + 1) * (num_cp + 1)).reshape(num_gt + 1, num_cp + 1)
            
            gt_areas = np.bincount(flat_gt, minlength=num_gt + 1)
            cp_areas = np.bincount(flat_cp, minlength=num_cp + 1)
            
            overlap_fg = overlap[1:, 1:]
            union_fg = gt_areas[1:, None] + cp_areas[None, 1:] - overlap_fg
            iou_matrix = overlap_fg / np.maximum(union_fg, 1)

            best_ious = iou_matrix.max(axis=1) if num_cp > 0 else np.zeros(num_gt)
        else:
            best_ious = np.zeros(num_gt)

        m03 = int((best_ious >= 0.3).sum())
        m05 = int((best_ious >= 0.5).sum())

        for val in best_ious:
            if val >= 0.3:
                all_ious.append(float(val))

        total_gt += num_gt
        total_cp += num_cp
        matched_03 += m03
        matched_05 += m05

        rec = (m05 / num_gt) * 100 if num_gt > 0 else 0
        print(f"[{idx}/{num_images}] {fn:<25} | {num_gt:<4} | {num_cp:<8} | {m03:<10} | {m05:<10} | {rec:6.1f}% | {elapsed:5.1f}s", flush=True)

    print("=" * 90, flush=True)
    recall_05 = (matched_05 / total_gt) * 100 if total_gt > 0 else 0
    recall_03 = (matched_03 / total_gt) * 100 if total_gt > 0 else 0
    prec_05 = (matched_05 / total_cp) * 100 if total_cp > 0 else 0
    prec_03 = (matched_03 / total_cp) * 100 if total_cp > 0 else 0
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0

    print(f"\nOVERALL CELLPOSE BENCHMARK SUMMARY ({num_images} Test Images, {total_gt} Manual GT Cells):", flush=True)
    print(f"  - Total Ground Truth Cells:        {total_gt}", flush=True)
    print(f"  - Total Cellpose Detected Cells:   {total_cp}", flush=True)
    print(f"  - Matched (IoU >= 0.5):            {matched_05} / {total_gt} (Recall: {recall_05:.1f}%)", flush=True)
    print(f"  - Matched (IoU >= 0.3):            {matched_03} / {total_gt} (Recall: {recall_03:.1f}%)", flush=True)
    print(f"  - Precision (IoU >= 0.5):          {prec_05:.1f}%", flush=True)
    print(f"  - Precision (IoU >= 0.3):          {prec_03:.1f}%", flush=True)
    print(f"  - Mean IoU (matched cells):        {mean_iou:.3f}", flush=True)
    print("=" * 90, flush=True)

    results_dict = {
        "num_test_images": num_images,
        "total_gt": total_gt,
        "total_cp": total_cp,
        "matched_05": matched_05,
        "matched_03": matched_03,
        "recall_05": round(recall_05, 2),
        "recall_03": round(recall_03, 2),
        "precision_05": round(prec_05, 2),
        "precision_03": round(prec_03, 2),
        "mean_iou": round(mean_iou, 4),
        "mask_rcnn_baseline": {
            "dice": 84.4,
            "iou": 73.0,
            "f1": 76.2
        }
    }

    res_path = os.path.join(root_dir, 'results', 'cellpose_msc_benchmark_results.json')
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\n[*] Results saved to: {res_path}", flush=True)

if __name__ == '__main__':
    run_cellpose_benchmark(10)
