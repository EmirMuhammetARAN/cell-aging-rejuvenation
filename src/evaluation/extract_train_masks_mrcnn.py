# -*- coding: utf-8 -*-
"""
TRAINING SET MASK R-CNN SEGMENTATION EXTRACTOR
=============================================
High-performance batch mask extractor running on NVIDIA RTX 4080 GPU via TF 2.10.
Extracts single-cell segmentation masks (.npy) for all training images:
- data/processed_v4/train/senescent (3,131 images)
- data/processed_v4/train/young (5,181 images)

Saves:
- scratch/mrcnn_masks_train_senescent.npy
- scratch/mrcnn_masks_train_young.npy
"""

import os
import sys
import time
import argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Enable GPU DLLs on Windows for TF 2.10
base_site = os.path.join(os.path.dirname(sys.executable), '..', 'Lib', 'site-packages')
nvidia_dir = os.path.abspath(os.path.join(base_site, 'nvidia'))
if os.path.exists(nvidia_dir):
    for root, dirs, files in os.walk(nvidia_dir):
        if any(f.endswith('.dll') for f in files):
            try:
                os.add_dll_directory(root)
                os.environ['PATH'] = root + ';' + os.environ['PATH']
            except Exception:
                pass

cuda11_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin'
if os.path.exists(cuda11_bin):
    try:
        os.add_dll_directory(cuda11_bin)
        os.environ['PATH'] = cuda11_bin + ';' + os.environ['PATH']
    except Exception:
        pass

import numpy as np
from PIL import Image

# 2. TF2 / Keras Compatibility Patch
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

# Ensure project root in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def extract_masks(model, img_dir, save_path, batch_save_interval=50):
    print(f"\n[*] Processing directory: {img_dir}", flush=True)
    files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))])
    N = len(files)
    print(f"    Total images found: {N}", flush=True)

    masks_dict = {}
    if os.path.exists(save_path):
        try:
            loaded = np.load(save_path, allow_pickle=True)
            masks_dict = loaded.item() if (hasattr(loaded, 'shape') and loaded.ndim == 0) else dict(loaded)
            print(f"    [RESUME] Loaded existing dictionary with {len(masks_dict)}/{N} masks.", flush=True)
        except Exception as e:
            print(f"    [WARN] Failed to load existing save_path: {e}. Starting fresh.", flush=True)
            masks_dict = {}

    to_process = [f for f in files if f not in masks_dict]
    print(f"    Images remaining to process: {len(to_process)}", flush=True)

    if not to_process:
        print("    [COMPLETE] All images already processed!", flush=True)
        return masks_dict

    start_time = time.time()
    processed_count = 0
    total_to_do = len(to_process)

    for idx, fname in enumerate(to_process):
        p_img = os.path.join(img_dir, fname)
        try:
            img_np = np.array(Image.open(p_img).convert('RGB'))
            results = model.detect([img_np], verbose=0)[0]
            masks = results['masks']
            if masks.shape[-1] > 0:
                combined = np.sum(masks, axis=-1) > 0
                masks_dict[fname] = combined
            else:
                masks_dict[fname] = np.zeros(img_np.shape[:2], dtype=bool)
        except Exception as err:
            print(f"    [ERROR] Processing {fname}: {err}", flush=True)
            masks_dict[fname] = np.zeros((512, 512), dtype=bool)

        processed_count += 1

        if processed_count % batch_save_interval == 0 or processed_count == total_to_do:
            elapsed = time.time() - start_time
            rate = processed_count / max(0.001, elapsed)
            remaining_sec = (total_to_do - processed_count) / max(0.001, rate)
            rem_min = remaining_sec / 60.0
            print(f"    [{processed_count}/{total_to_do}] ({len(masks_dict)}/{N} total - {100.0*len(masks_dict)/N:.1f}%) "
                  f"| Rate: {rate:.2f} img/s | ETA: {rem_min:.1f} min | Checkpoint saved.", flush=True)
            np.save(save_path, masks_dict)

    np.save(save_path, masks_dict)
    print(f"[OK] Successfully saved {len(masks_dict)} masks to {save_path}", flush=True)
    return masks_dict


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN Training Dataset Mask Extractor")
    parser.add_argument('--category', choices=['senescent', 'young', 'both'], default='both',
                        help="Which train category to process: senescent, young, or both")
    args = parser.parse_args()

    gpus = tf.config.list_physical_devices('GPU')
    print(f"[*] TensorFlow GPU Devices available: {gpus}", flush=True)

    weights_path = os.path.join(root_dir, "fatma hoca", "mask_rcnn_object_0800.h5")
    scratch_dir = os.path.join(root_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    print(f"[*] Loading Mask R-CNN model weights from: {weights_path}", flush=True)
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.path.dirname(weights_path))
    model.load_weights(weights_path, by_name=True)
    print("[*] Model weights successfully loaded into GPU VRAM.", flush=True)

    # 1. Process Senescent training set first (3,131 images, resume from 100)
    if args.category in ['senescent', 'both']:
        sen_dir = os.path.join(root_dir, "data", "processed_v4", "train", "senescent")
        save_sen = os.path.join(scratch_dir, "mrcnn_masks_train_senescent.npy")
        extract_masks(model, sen_dir, save_sen)

    # 2. Process Young training set next (5,181 images)
    if args.category in ['young', 'both']:
        young_dir = os.path.join(root_dir, "data", "processed_v4", "train", "young")
        save_young = os.path.join(scratch_dir, "mrcnn_masks_train_young.npy")
        extract_masks(model, young_dir, save_young)

    print("\n==========================================", flush=True)
    print("[FINISHED] All training set masks extracted!", flush=True)
    print("==========================================", flush=True)


if __name__ == '__main__':
    main()
