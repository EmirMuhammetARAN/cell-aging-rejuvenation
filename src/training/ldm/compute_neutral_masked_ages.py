# -*- coding: utf-8 -*-
"""
COMPUTE CONTINUOUS BIOLOGICAL AGES VIA CLASSIFIER V3 WITH NEUTRAL GRAY MASKING
=============================================================================
Loads Mask R-CNN masks (scratch/mrcnn_masks_train_young.npy and
scratch/mrcnn_masks_train_senescent.npy), applies Neutral Gray (pixel 128)
background masking to eliminate non-cellular debris and culture artifacts,
and passes cells through Classifier v3 (ConvNeXt-Tiny) to obtain true biological
lifespan scores [0.0, 1.0] and passage ages [P1 - P41].

Saves:
  - data/processed_v4/train_continuous_ages_neutral_masked.pt
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# Ensure root dir in path
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

class ConvNeXtMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[2].in_features
        self.norm = base.classifier[0]
        self.age_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.phase_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        x = self.norm(self.avgpool(self.features(x)))
        return self.age_head(x).squeeze(1), self.phase_head(x)


def apply_neutral_gray_mask(img_pil, mask_bool):
    """Fills background pixels outside cell mask with neutral gray (128)."""
    gray = img_pil.convert('L')
    rgb = Image.merge('RGB', (gray, gray, gray))
    if mask_bool is None or mask_bool.sum() == 0:
        return rgb # Fallback to original image if no mask detected
    
    np_img = np.array(rgb)
    out_np = np_img.copy()
    out_np[~mask_bool] = 128
    return Image.fromarray(out_np)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running on device: {device}", flush=True)

    ckpt_path = os.path.join(root_dir, 'deneme', 'checkpoints', 'classifier_v3_convnext_best.pth')
    print(f"[*] Loading Classifier v3 model from: {ckpt_path}", flush=True)
    model = ConvNeXtMultiTask().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Mask R-CNN masks
    scratch_dir = os.path.join(root_dir, 'scratch')
    p_young_masks = os.path.join(scratch_dir, 'mrcnn_masks_train_young.npy')
    p_senes_masks = os.path.join(scratch_dir, 'mrcnn_masks_train_senescent.npy')

    if not os.path.exists(p_young_masks) or not os.path.exists(p_senes_masks):
        print(f"[ERROR] Masks not found! Ensure both {p_young_masks} and {p_senes_masks} exist.")
        sys.exit(1)

    print("[*] Loading Mask R-CNN masks dictionaries...", flush=True)
    masks_young = np.load(p_young_masks, allow_pickle=True)
    masks_young = masks_young.item() if hasattr(masks_young, 'item') else dict(masks_young)
    masks_senes = np.load(p_senes_masks, allow_pickle=True)
    masks_senes = masks_senes.item() if hasattr(masks_senes, 'item') else dict(masks_senes)
    print(f"    Loaded {len(masks_young)} young masks, {len(masks_senes)} senescent masks.", flush=True)

    train_young_dir = os.path.join(root_dir, 'data', 'processed_v4', 'train', 'young')
    train_senes_dir = os.path.join(root_dir, 'data', 'processed_v4', 'train', 'senescent')

    items = []
    for fname in sorted(os.listdir(train_young_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            p = os.path.join(train_young_dir, fname)
            items.append((p, 0, fname, masks_young.get(fname, None)))

    for fname in sorted(os.listdir(train_senes_dir)):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            p = os.path.join(train_senes_dir, fname)
            items.append((p, 1, fname, masks_senes.get(fname, None)))

    print(f"[*] Total images to score: {len(items)} (Young: {len(masks_young)}, Senescent: {len(masks_senes)})", flush=True)

    scores_dict = {}
    young_scores = []
    senes_scores = []

    batch_size = 64
    t0 = time.time()
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i+batch_size]
        tensors = []
        for p, label, fname, mask in batch_items:
            with Image.open(p) as img:
                masked_pil = apply_neutral_gray_mask(img, mask)
            tensors.append(val_transform(masked_pil))
        
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            preds, _ = model(batch_tensor)
            preds_np = preds.cpu().numpy()

        for (p, label, fname, mask), score in zip(batch_items, preds_np):
            rel_path = os.path.relpath(p, root_dir)
            norm_rel = rel_path.replace('\\', '/')
            scores_dict[fname] = float(score)
            scores_dict[rel_path] = float(score)
            scores_dict[norm_rel] = float(score)

            if label == 0:
                young_scores.append(score)
            else:
                senes_scores.append(score)

        if (i // batch_size) % 10 == 0 or (i + batch_size) >= len(items):
            print(f"    [{min(i + batch_size, len(items))}/{len(items)}] Scored ({((min(i + batch_size, len(items))) / (time.time() - t0)):.1f} img/s)...", flush=True)

    y_arr = np.array(young_scores)
    s_arr = np.array(senes_scores)
    y_passages = y_arr * 40.0 + 1.0
    s_passages = s_arr * 40.0 + 1.0

    print(f"\n{'='*70}", flush=True)
    print("[CLASSIFIER V3 NEUTRAL-GRAY TRAINING COHORT BIOMARKERS SUMMARY]", flush=True)
    print(f"  Young Cells (N={len(y_arr)}):     Mean Score = {y_arr.mean():.4f} +/- {y_arr.std():.4f} | Mean Biological Age = P{y_passages.mean():.2f} +/- {y_passages.std():.2f}")
    print(f"  Senescent Cells (N={len(s_arr)}): Mean Score = {s_arr.mean():.4f} +/- {s_arr.std():.4f} | Mean Biological Age = P{s_passages.mean():.2f} +/- {s_passages.std():.2f}")
    print(f"  Biological Age Separation (Delta P): {s_passages.mean() - y_passages.mean():+.2f} Passages")
    print(f"{'='*70}", flush=True)

    save_path = os.path.join(root_dir, 'data', 'processed_v4', 'train_continuous_ages_neutral_masked.pt')
    torch.save(scores_dict, save_path)
    print(f"[OK] Saved continuous age mappings to: {save_path}", flush=True)


if __name__ == '__main__':
    main()
