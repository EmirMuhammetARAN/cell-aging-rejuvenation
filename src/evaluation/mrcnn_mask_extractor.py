# -*- coding: utf-8 -*-
"""
MASK R-CNN SINGLE-CELL SEGMENTATION EXTRACTOR
=============================================
Unified extractor for generating Ground-Truth single-cell segmentation masks (.npy)
for both LDM (v12) and CycleGAN (Epoch 160) models using TensorFlow 2.10.

Usage:
  python mrcnn_mask_extractor.py --target ldm
  python mrcnn_mask_extractor.py --target cyclegan
  python mrcnn_mask_extractor.py --target all
"""

import os
import sys
import time
import argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from PIL import Image

# TF2 Compatibility Patch
import tensorflow as tf
import keras.saving.hdf5_format as hdf5_format
import keras.engine.saving as saving

class LayerListWrapper:
    def __init__(self, layers):
        self.layers = list(layers)
        self._trainable_weights = []
        self._non_trainable_weights = []
    def _flatten_layers(self, recursive=True, include_self=True):
        return self.layers

def patched_load_by_name(f, layers, skip_mismatch=True):
    wrapper = LayerListWrapper(layers)
    hdf5_format.load_weights_from_hdf5_group_by_name(f, wrapper, skip_mismatch=skip_mismatch)

saving.load_weights_from_hdf5_group_by_name = patched_load_by_name

# Ensure project root is in sys.path
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from mrcnn.config import Config
from mrcnn import model as modellib


class InferenceConfig(Config):
    NAME = 'object'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2
    DETECTION_MIN_CONFIDENCE = 0.5


def extract_masks_for_dataset(model, img_dir, save_path):
    print(f"\n[*] Processing directory: {img_dir}")
    files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    N = len(files)
    print(f"    Found {N} test images.")
    
    if os.path.exists(save_path):
        loaded = np.load(save_path, allow_pickle=True)
        masks_dict = loaded.item() if (hasattr(loaded, 'shape') and loaded.ndim == 0) else dict(loaded)
        print(f"    [RESUME] Loaded existing dictionary with {len(masks_dict)} masks")
    else:
        masks_dict = {}
        
    start_time = time.time()
    for idx, fname in enumerate(files):
        if fname in masks_dict:
            continue
            
        p_img = os.path.join(img_dir, fname)
        img_np = np.array(Image.open(p_img).convert('RGB'))
        results = model.detect([img_np], verbose=0)[0]
        masks = results['masks']
        
        if masks.shape[-1] > 0:
            combined = np.sum(masks, axis=-1) > 0
            masks_dict[fname] = combined
        else:
            masks_dict[fname] = np.zeros(img_np.shape[:2], dtype=bool)
            
        if (idx + 1) % 50 == 0 or (idx + 1) == N:
            elapsed = time.time() - start_time
            rate = (idx + 1) / max(1, elapsed)
            print(f"    [{idx+1}/{N}] Processed ({rate:.2f} img/s)...")
            np.save(save_path, masks_dict)
            
    np.save(save_path, masks_dict)
    print(f"[OK] Saved {len(masks_dict)} masks to {save_path}")
    return masks_dict


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN Single-Cell Mask Extractor")
    parser.add_argument('--target', choices=['ldm', 'cyclegan', 'all'], default='all',
                        help="Target dataset to extract: ldm, cyclegan, or all")
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    weights_path = os.path.join(base_dir, "fatma hoca", "mask_rcnn_object_0800.h5")
    scratch_dir = os.path.join(base_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    
    print("[*] Loading Mask R-CNN weights...")
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.dirname(weights_path))
    model.load_weights(weights_path, by_name=True)
    
    # Official evaluated datasets: LDM Seed 2026 (Best FID) & CycleGAN Epoch 160
    if args.target in ['ldm', 'all']:
        aging_dir = os.path.join(base_dir, "results", "generated", "seed_sweep", "seed_2026", "aging")
        reju_dir = os.path.join(base_dir, "results", "generated", "seed_sweep", "seed_2026", "rejuv")
        extract_masks_for_dataset(model, aging_dir, os.path.join(scratch_dir, "mrcnn_masks_aging.npy"))
        extract_masks_for_dataset(model, reju_dir, os.path.join(scratch_dir, "mrcnn_masks_reju.npy"))
        
    if args.target in ['cyclegan', 'all']:
        cg_aging_dir = os.path.join(base_dir, "results", "generated", "cyclegan_v2_sweep", "epoch_160", "aging")
        cg_reju_dir = os.path.join(base_dir, "results", "generated", "cyclegan_v2_sweep", "epoch_160", "rejuvenation")
        extract_masks_for_dataset(model, cg_aging_dir, os.path.join(scratch_dir, "mrcnn_masks_cyclegan_aging.npy"))
        extract_masks_for_dataset(model, cg_reju_dir, os.path.join(scratch_dir, "mrcnn_masks_cyclegan_reju.npy"))
        
    print("\n[FINISHED] Mask extraction complete!")


if __name__ == '__main__':
    main()
