"""
Dataset for LDM v16 Stage 1: Fibroblast tiles (Columbia Picard lifespan study) with REAL passage labels.

Source: deneme/processed_tiles/{images/, metadata.csv}
  metadata columns: tile_filename, donor, passage, normalized_age, age_group, row, col, source_image

Split policy (donor-level, never tile-level -> no leakage between the 20 tiles of one source image):
  * val   = held-out donor (default hFB14, same donor classifier-v3 used for validation)
  * train = all other donors

Returns (image [-1,1] RGB, passage float, cell_type long) -> cell_type 0 = Fibroblast.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF

CELL_FIBROBLAST = 0
CELL_MSC = 1


class FibroblastPassageDataset(Dataset):
    def __init__(self, root_dir, split='train', val_donor='hFB14',
                 tiles_dir=None, augment=None, passage_jitter=0.25):
        self.split = split
        self.is_train = (split == 'train')
        self.augment = self.is_train if augment is None else augment
        self.passage_jitter = passage_jitter if self.is_train else 0.0

        tiles_dir = tiles_dir or os.path.join(root_dir, 'deneme', 'processed_tiles')
        self.img_dir = os.path.join(tiles_dir, 'images')
        df = pd.read_csv(os.path.join(tiles_dir, 'metadata.csv'))

        if split == 'train':
            df = df[df['donor'] != val_donor]
        elif split == 'val':
            df = df[df['donor'] == val_donor]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        self.df = df.reset_index(drop=True)

        self.fnames = self.df['tile_filename'].tolist()
        self.passages = self.df['passage'].astype(np.float32).to_numpy()
        self.donors = self.df['donor'].tolist()
        print(f"[FibroblastPassageDataset] {split}: {len(self.df)} tiles | "
              f"donors={sorted(set(self.donors))} | passage {int(self.passages.min())}-{int(self.passages.max())}")

    def __len__(self):
        return len(self.df)

    def make_sampler(self, seed=2026, power=0.5):
        """Passage-balanced sampler: weight ~ 1 / count(passage)^power (power=0.5 -> inverse sqrt)."""
        uniq, inv, counts = np.unique(self.passages, return_inverse=True, return_counts=True)
        w = 1.0 / np.power(counts[inv].astype(np.float64), power)
        gen = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(self), replacement=True, generator=gen)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.fnames[idx])).convert('RGB')
        passage = float(self.passages[idx])

        if self.augment:  # square tiles -> flips + rot90 are exact symmetries
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            k = random.randint(0, 3)
            if k:
                img = img.rotate(90 * k)
        if self.passage_jitter > 0:
            passage += random.uniform(-self.passage_jitter, self.passage_jitter)

        t = TF.to_tensor(img)
        t = TF.normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return t, torch.tensor(passage, dtype=torch.float32), torch.tensor(CELL_FIBROBLAST, dtype=torch.long)


class MSCPassageDataset(Dataset):
    """
    Dataset for LDM v16 Stage 2: MSC crops (data/processed_v4) with class-calibrated continuous passages.
    - Split policy (FOV-level, strictly independent of data/processed_v4/test):
        * 'train' = 7,696 MSC crops from data/processed_v4/train excluding data/raw/val FOVs
        * 'val'   = 616 MSC crops from data/processed_v4/train belonging to the 20 held-out FOVs in data/raw/val
        * 'test'  = 645 MSC crops from data/processed_v4/test (blinded; never used in training or checkpoint selection)
    - Preserves ground-truth class boundaries (0% cross-class label contamination) while maintaining
      the continuous intra-class biological age ordering predicted by Classifier v3 (ConvNeXt-Tiny):
        * Young MSCs (class 0):     mapped monotonically to P[2.0, 8.5]  (Median P3.5)
        * Senescent MSCs (class 1): mapped monotonically to P[8.5, 20.0] (Median P14.5)
      Validation/test passages are calibrated strictly via monotonic interpolation from the 'train' split quantiles.
    - Returns (image [-1, 1] RGB, passage float, cell_type=1 [MSC])
    """
    def __init__(self, root_dir, split='train', data_version='processed_v4',
                 augment=None, passage_jitter=0.20, calibrate_classes=True):
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")
        self.root_dir = root_dir
        self.split = split
        self.is_train = (split == 'train')
        self.augment = self.is_train if augment is None else augment
        self.passage_jitter = passage_jitter if self.is_train else 0.0
        self.calibrate_classes = calibrate_classes

        val_raw_dir = os.path.join(root_dir, 'data', 'raw', 'val')
        val_raw_prefixes = sorted([
            os.path.splitext(f)[0] for f in os.listdir(val_raw_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        ]) if os.path.exists(val_raw_dir) else []

        cache_path = os.path.join(root_dir, 'data', data_version, 'train_continuous_ages_neutral_masked.pt')
        if not os.path.exists(cache_path):
            cache_path = os.path.join(root_dir, 'data', data_version, 'train_continuous_ages.pt')
        age_cache = torch.load(cache_path, map_location='cpu') if os.path.exists(cache_path) else {}

        # Always collect train_pool first so val/test calibration is fit strictly on train_pool
        train_pool = []
        val_pool = []
        train_data_dir = os.path.join(root_dir, 'data', data_version, 'train')
        for cls_idx, (class_name, def_score) in enumerate([('young', 0.0625), ('senescent', 0.3625)]):
            cdir = os.path.join(train_data_dir, class_name)
            if not os.path.exists(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    abs_p = os.path.join(cdir, fname)
                    rel_p = os.path.relpath(abs_p, root_dir)
                    norm_p = rel_p.replace('\\', '/')
                    s = float(age_cache.get(fname, age_cache.get(rel_p, age_cache.get(norm_p, def_score))))
                    raw_pas = s * 40.0 + 1.0
                    rec = (abs_p, fname, raw_pas, cls_idx)
                    if any(fname.startswith(v + '_') for v in val_raw_prefixes):
                        val_pool.append(rec)
                    else:
                        train_pool.append(rec)

        if split == 'train':
            raw_records = train_pool
        elif split == 'val':
            raw_records = val_pool
        else:
            pred_csv_path = os.path.join(root_dir, 'deneme', 'convnext_msc_test_predictions.csv')
            pred_map = {}
            if os.path.exists(pred_csv_path):
                df_pred = pd.read_csv(pred_csv_path)
                pred_map = dict(zip(df_pred['filename'], df_pred['predicted_eq_passage']))

            raw_records = []
            test_data_dir = os.path.join(root_dir, 'data', data_version, 'test')
            for cls_idx, (class_name, def_pas) in enumerate([('young', 3.5), ('senescent', 14.5)]):
                cdir = os.path.join(test_data_dir, class_name)
                if not os.path.exists(cdir):
                    continue
                for fname in sorted(os.listdir(cdir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        abs_p = os.path.join(cdir, fname)
                        raw_pas = float(pred_map.get(fname, def_pas))
                        raw_records.append((abs_p, fname, raw_pas, cls_idx))

        raw_passages = np.array([r[2] for r in raw_records], dtype=np.float32)
        class_labels = np.array([r[3] for r in raw_records], dtype=np.int64)

        if self.calibrate_classes and len(raw_records) > 0:
            # Fit rank calibration strictly on train_pool, then apply to target split
            tr_raw = np.array([r[2] for r in train_pool], dtype=np.float32)
            tr_cls = np.array([r[3] for r in train_pool], dtype=np.int64)

            cal_passages = np.copy(raw_passages)
            for cls_idx, (p_min, p_med, p_max) in [
                (0, (2.0, 3.5, 8.5)),
                (1, (8.5, 14.5, 20.0)),
            ]:
                mask_target = (class_labels == cls_idx)
                mask_tr = (tr_cls == cls_idx)
                if not np.any(mask_target) or mask_tr.sum() <= 1:
                    continue
                tr_vals = tr_raw[mask_tr]
                tr_cal = self._rank_calibrate(tr_vals, p_min=p_min, p_med=p_med, p_max=p_max)
                if split == 'train':
                    cal_passages[mask_target] = tr_cal
                else:
                    order = np.argsort(tr_vals, kind='mergesort')
                    cal_passages[mask_target] = np.interp(
                        raw_passages[mask_target],
                        tr_vals[order],
                        tr_cal[order],
                        left=p_min,
                        right=p_max,
                    ).astype(np.float32)
            self.passages = cal_passages
        else:
            self.passages = raw_passages

        self.class_labels = class_labels
        self.samples = [
            (r[0], r[1], float(self.passages[i]))
            for i, r in enumerate(raw_records)
        ]

        y_p = self.passages[self.class_labels == 0]
        s_p = self.passages[self.class_labels == 1]
        print(f"[MSCPassageDataset] {split}: {len(self.samples)} images | "
              f"Young(n={len(y_p)}): P{y_p.min():.1f}-{y_p.max():.1f} (med {np.median(y_p):.1f}) | "
              f"Senescent(n={len(s_p)}): P{s_p.min():.1f}-{s_p.max():.1f} (med {np.median(s_p):.1f})")

    @staticmethod
    def _rank_calibrate(vals, p_min, p_med, p_max):
        """Maps continuous scores monotonically by empirical quantile onto [p_min, p_max] with exact median p_med."""
        n = len(vals)
        order = np.argsort(vals, kind='mergesort')
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, n, dtype=np.float32)
        out = np.where(
            ranks <= 0.5,
            p_min + (ranks / 0.5) * (p_med - p_min),
            p_med + ((ranks - 0.5) / 0.5) * (p_max - p_med)
        )
        return out.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def make_sampler(self, seed=2026, power=0.0, bin_size=1.0):
        """
        50/50 Class-Balanced Sampler.
        Since _rank_calibrate already distributes cells monotonically across the passage range by empirical quantile,
        power=0.0 gives exact 50/50 class balance with uniform weight per cell (avoiding boundary bin distortion).
        """
        w = np.zeros(len(self.samples), dtype=np.float64)
        for cls_idx in (0, 1):
            mask = (self.class_labels == cls_idx)
            if not np.any(mask):
                continue
            if power <= 0.0:
                cls_w = np.ones(mask.sum(), dtype=np.float64)
            else:
                cls_pas = self.passages[mask]
                p_min = float(cls_pas.min())
                binned = np.floor((cls_pas - p_min) / bin_size).astype(np.int64)
                _, inv, counts = np.unique(binned, return_inverse=True, return_counts=True)
                cls_w = 1.0 / np.power(counts[inv].astype(np.float64), power)
            cls_w = cls_w / cls_w.sum() * 0.5
            w[mask] = cls_w
        gen = torch.Generator().manual_seed(seed)
        return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(self), replacement=True, generator=gen)

    def __getitem__(self, idx):
        path, fname, passage = self.samples[idx]
        img = Image.open(path).convert('RGB')

        if self.augment:
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            k = random.randint(0, 3)
            if k:
                img = img.rotate(90 * k)
        if self.passage_jitter > 0:
            passage = passage + random.uniform(-self.passage_jitter, self.passage_jitter)

        t = TF.to_tensor(img)
        t = TF.normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return t, torch.tensor(passage, dtype=torch.float32), torch.tensor(CELL_MSC, dtype=torch.long)
