"""Extract Mask R-CNN segmentation masks for data/processed_v5 train and val sets.

Pre-populates matching masks via filename/MD5 from archived v4 mask caches before
running Mask R-CNN inference on remaining uncached images.
"""

import argparse
import hashlib
import os
import sys
import time

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from PIL import Image

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

VALID_EXTS = (".jpg", ".jpeg", ".png")


def _enable_windows_cuda_dlls():
    base_site = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages")
    nvidia_dir = os.path.abspath(os.path.join(base_site, "nvidia"))
    if os.path.exists(nvidia_dir):
        for root, _, files in os.walk(nvidia_dir):
            if any(f.endswith(".dll") for f in files):
                try:
                    os.add_dll_directory(root)
                    os.environ["PATH"] = root + ";" + os.environ["PATH"]
                except Exception:
                    pass

    cuda11_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
    if os.path.exists(cuda11_bin):
        try:
            os.add_dll_directory(cuda11_bin)
            os.environ["PATH"] = cuda11_bin + ";" + os.environ["PATH"]
        except Exception:
            pass


def _file_md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def _load_mask_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    loaded = np.load(path, allow_pickle=True)
    return loaded.item() if (hasattr(loaded, "shape") and loaded.ndim == 0) else dict(loaded)


def build_archive_mask_lookup(base_dir: str):
    """Build filename and MD5 mask lookups from archived Mask R-CNN caches."""
    arch_masks_dir = os.path.join(base_dir, "scratch", "archive")
    mask_names = [
        "mrcnn_masks_train_young.npy",
        "mrcnn_masks_train_senescent.npy",
        "mrcnn_masks_input_young.npy",
        "mrcnn_masks_input_senescent.npy",
    ]
    by_name = {}
    for fn in mask_names:
        p = os.path.join(arch_masks_dir, fn)
        if os.path.exists(p):
            try:
                by_name.update(_load_mask_dict(p))
            except Exception as err:
                print(f"[WARN] Failed loading archived masks {fn}: {err}", flush=True)

    by_md5 = {}
    v4_dir = os.path.join(base_dir, "data", "archive", "processed_v4")
    if by_name and os.path.exists(v4_dir):
        for root, _, files in os.walk(v4_dir):
            for fname in files:
                if fname in by_name and fname.lower().endswith(VALID_EXTS):
                    digest = _file_md5(os.path.join(root, fname))
                    by_md5.setdefault(digest, by_name[fname])

    print(
        f"[*] Archive lookup ready: {len(by_name)} named masks, {len(by_md5)} MD5-indexed masks.",
        flush=True,
    )
    return by_name, by_md5


def load_mrcnn_model(weights_path: str):
    """Configure TF 2.10 compatibility patch and load Mask R-CNN inference model."""
    _enable_windows_cuda_dlls()
    import keras.engine.saving as saving
    import keras.saving.hdf5_format as hdf5_format
    import tensorflow as tf
    from mrcnn import model as modellib
    from mrcnn.config import Config

    class LayerListWrapper:
        def __init__(self, layers):
            self.layers = list(layers)
            self._trainable_weights = []
            self._non_trainable_weights = []

        def _flatten_layers(self, recursive=True, include_self=True):
            return self.layers

    def patched_load_by_name(f, layers, skip_mismatch=True):
        hdf5_format.load_weights_from_hdf5_group_by_name(
            f, LayerListWrapper(layers), skip_mismatch=skip_mismatch
        )

    saving.load_weights_from_hdf5_group_by_name = patched_load_by_name

    class InferenceConfig(Config):
        NAME = "object"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 2
        DETECTION_MIN_CONFIDENCE = 0.5

    gpus = tf.config.list_physical_devices("GPU")
    print(f"[*] TensorFlow GPU Devices: {gpus}", flush=True)
    model = modellib.MaskRCNN(
        mode="inference", config=InferenceConfig(), model_dir=os.path.dirname(weights_path)
    )
    model.load_weights(weights_path, by_name=True)
    print(f"[*] Loaded Mask R-CNN weights from {weights_path}", flush=True)
    return model


def prepare_target_masks(img_dir: str, save_path: str, arch_by_name: dict, arch_by_md5: dict):
    """Load existing save_path, backfill archive matches, and return missing filenames."""
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(VALID_EXTS))
    masks_dict = _load_mask_dict(save_path)
    reused = 0
    for fname in files:
        if fname in masks_dict:
            continue
        if fname in arch_by_name:
            masks_dict[fname] = np.asarray(arch_by_name[fname], dtype=bool)
            reused += 1
        else:
            digest = _file_md5(os.path.join(img_dir, fname))
            if digest in arch_by_md5:
                masks_dict[fname] = np.asarray(arch_by_md5[digest], dtype=bool)
                reused += 1

    if reused > 0:
        np.save(save_path, masks_dict)

    to_process = [f for f in files if f not in masks_dict]
    print(
        f"[*] {os.path.relpath(img_dir, root_dir)}: total={len(files)} | "
        f"cached={len(masks_dict)} (+{reused} from archive) | remaining={len(to_process)}",
        flush=True,
    )
    return files, masks_dict, to_process


def extract_remaining(model, img_dir: str, save_path: str, masks_dict: dict, to_process: list, total_n: int, batch_save_interval: int = 50):
    if not to_process:
        return masks_dict

    t0 = time.time()
    for idx, fname in enumerate(to_process, 1):
        img_path = os.path.join(img_dir, fname)
        try:
            img_np = np.array(Image.open(img_path).convert("RGB"))
            masks = model.detect([img_np], verbose=0)[0]["masks"]
            if masks.shape[-1] > 0:
                masks_dict[fname] = np.sum(masks, axis=-1) > 0
            else:
                masks_dict[fname] = np.zeros(img_np.shape[:2], dtype=bool)
        except Exception as err:
            print(f"    [ERROR] Processing {fname}: {err}", flush=True)
            masks_dict[fname] = np.zeros((512, 512), dtype=bool)

        if idx % batch_save_interval == 0 or idx == len(to_process):
            rate = idx / max(1e-3, time.time() - t0)
            eta_min = (len(to_process) - idx) / max(1e-3, rate) / 60.0
            print(
                f"    [{idx}/{len(to_process)}] ({len(masks_dict)}/{total_n}) | "
                f"{rate:.2f} img/s | ETA: {eta_min:.1f} min",
                flush=True,
            )
            np.save(save_path, masks_dict)

    print(f"[OK] Saved {len(masks_dict)} masks to {save_path}", flush=True)
    return masks_dict


def main():
    parser = argparse.ArgumentParser(description="Mask R-CNN Train & Val Mask Extractor for processed_v5")
    parser.add_argument("--category", choices=["senescent", "young", "both"], default="both")
    args = parser.parse_args()

    categories = ["senescent", "young"] if args.category == "both" else [args.category]
    scratch_dir = os.path.join(root_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)

    arch_by_name, arch_by_md5 = build_archive_mask_lookup(root_dir)

    jobs = []
    for split in ("train", "val"):
        for cat in categories:
            img_dir = os.path.join(root_dir, "data", "processed_v5", split, cat)
            save_path = os.path.join(scratch_dir, f"mrcnn_masks_{split}_{cat}.npy")
            files, masks_dict, to_process = prepare_target_masks(img_dir, save_path, arch_by_name, arch_by_md5)
            jobs.append((img_dir, save_path, len(files), masks_dict, to_process))

    if any(job[4] for job in jobs):
        weights_path = os.path.join(root_dir, "fatma hoca", "mask_rcnn_object_0800.h5")
        model = load_mrcnn_model(weights_path)
        for img_dir, save_path, total_n, masks_dict, to_process in jobs:
            extract_remaining(model, img_dir, save_path, masks_dict, to_process, total_n)

    print("[FINISHED] All train/val masks ready.", flush=True)


if __name__ == "__main__":
    main()
