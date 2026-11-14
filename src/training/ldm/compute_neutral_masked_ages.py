"""Compute continuous biological passage scores on processed_v5 train and val splits.

Applies neutral gray (128) background masking from scratch/mrcnn_masks_*.npy
and scores crops with PassageClock v3.5, saving results to
data/processed_v5/train_continuous_ages_neutral_masked.pt.
"""

import os
import sys
import time
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.classifier.passage_clock import get_default_v3_5_ckpt, load_passage_clock, predict_passages


def apply_neutral_gray_mask(img_pil, mask_bool):
    """Fill background pixels outside the cell mask with neutral gray (128)."""
    gray = img_pil.convert("L")
    rgb = Image.merge("RGB", (gray, gray, gray))
    if mask_bool is None or mask_bool.sum() == 0:
        return rgb

    np_img = np.array(rgb)
    out_np = np_img.copy()
    out_np[~mask_bool] = 128
    return Image.fromarray(out_np)


def load_masks_dict(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[WARN] Mask file missing, falling back to unmasked images: {path}", flush=True)
        return {}
    loaded = np.load(path, allow_pickle=True)
    return loaded.item() if (hasattr(loaded, "shape") and loaded.ndim == 0) else dict(loaded)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}", flush=True)

    ckpt_path = get_default_v3_5_ckpt(root_dir)
    print(f"[*] Loading PassageClock v3.5 from: {ckpt_path}", flush=True)
    model, ckpt_meta = load_passage_clock(ckpt_path, device=device)

    scratch_dir = os.path.join(root_dir, "scratch")
    items = []
    for split in ("train", "val"):
        for cls_name, label in (("young", 0), ("senescent", 1)):
            mask_path = os.path.join(scratch_dir, f"mrcnn_masks_{split}_{cls_name}.npy")
            masks = load_masks_dict(mask_path)
            img_dir = os.path.join(root_dir, "data", "processed_v5", split, cls_name)
            if not os.path.exists(img_dir):
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    p = os.path.join(img_dir, fname)
                    items.append((split, p, label, fname, masks.get(fname, None)))

    print(f"[*] Total train+val images to score: {len(items)}", flush=True)

    scores_dict = {}
    stats = {
        ("train", 0): [],
        ("train", 1): [],
        ("val", 0): [],
        ("val", 1): [],
    }

    batch_size = 64
    t0 = time.time()
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        tensors = []
        for _, p, _, _, mask in batch_items:
            with Image.open(p) as img:
                masked_pil = apply_neutral_gray_mask(img, mask)
            tensors.append(TF.to_tensor(masked_pil))

        batch_tensor = torch.stack(tensors)
        passages_np, raw_np, _ = predict_passages(
            model, batch_tensor, device=device, bs=batch_size, meta=ckpt_meta, return_raw=True
        )

        for (split, p, label, fname, _), score, pas in zip(batch_items, raw_np, passages_np):
            rel_path = os.path.relpath(p, root_dir)
            norm_rel = rel_path.replace("\\", "/")
            scores_dict[fname] = float(score)
            scores_dict[rel_path] = float(score)
            scores_dict[norm_rel] = float(score)
            stats[(split, label)].append((float(score), float(pas)))

        done = min(i + batch_size, len(items))
        if (i // batch_size) % 15 == 0 or done == len(items):
            rate = done / max(1e-3, time.time() - t0)
            print(f"    [{done}/{len(items)}] Scored ({rate:.1f} img/s)...", flush=True)

    for split in ("train", "val"):
        y_vals = np.array(stats[(split, 0)])
        s_vals = np.array(stats[(split, 1)])
        if len(y_vals) and len(s_vals):
            print(
                f"[{split.upper()}] Young (N={len(y_vals)}): P{y_vals[:, 1].mean():.2f}+/-{y_vals[:, 1].std():.2f} | "
                f"Senescent (N={len(s_vals)}): P{s_vals[:, 1].mean():.2f}+/-{s_vals[:, 1].std():.2f} | "
                f"Delta={s_vals[:, 1].mean() - y_vals[:, 1].mean():+.2f}P",
                flush=True,
            )

    save_path = os.path.join(root_dir, "data", "processed_v5", "train_continuous_ages_neutral_masked.pt")
    torch.save(scores_dict, save_path)
    print(f"[OK] Saved {len(items)} continuous age mappings to: {save_path}", flush=True)


if __name__ == "__main__":
    main()
