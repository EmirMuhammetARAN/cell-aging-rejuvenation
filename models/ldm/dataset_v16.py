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
    Dataset for LDM v16 Stage 2: MSC crops (data/processed_v4) with continuous pseudo-passage labels.
    - Train split: 8,312 crops (young=5181, senescent=3131).
      Continuous scores loaded from data/processed_v4/train_continuous_ages_neutral_masked.pt
      Converted to passage: passage = score * 40.0 + 1.0 (range ~ [2.0, 38.5])
    - Val split: 645 test crops (young=327, senescent=318).
      Predictions loaded from deneme/convnext_msc_test_predictions.csv (predicted_eq_passage)
    - Returns (image [-1, 1] RGB, passage float, cell_type=1 [MSC])
    """
    def __init__(self, root_dir, split='train', data_version='processed_v4',
                 augment=None, passage_jitter=0.25):
        self.root_dir = root_dir
        self.split = split
        self.is_train = (split == 'train')
        self.augment = self.is_train if augment is None else augment
        self.passage_jitter = passage_jitter if self.is_train else 0.0

        self.samples = []
        if self.is_train:
            cache_path = os.path.join(root_dir, 'data', data_version, 'train_continuous_ages_neutral_masked.pt')
            if not os.path.exists(cache_path):
                cache_path = os.path.join(root_dir, 'data', data_version, 'train_continuous_ages.pt')
            age_cache = torch.load(cache_path, map_location='cpu') if os.path.exists(cache_path) else {}

            data_dir = os.path.join(root_dir, 'data', data_version, 'train')
            for class_name, def_score in [('young', 0.15), ('senescent', 0.65)]:
                cdir = os.path.join(data_dir, class_name)
                if not os.path.exists(cdir):
                    continue
                for fname in sorted(os.listdir(cdir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        abs_p = os.path.join(cdir, fname)
                        rel_p = os.path.relpath(abs_p, root_dir)
                        norm_p = rel_p.replace('\\', '/')
                        s = float(age_cache.get(fname, age_cache.get(rel_p, age_cache.get(norm_p, def_score))))
                        pas = s * 40.0 + 1.0
                        self.samples.append((abs_p, fname, pas))
        else:
            # Test split (645 images)
            pred_csv_path = os.path.join(root_dir, 'deneme', 'convnext_msc_test_predictions.csv')
            pred_map = {}
            if os.path.exists(pred_csv_path):
                df_pred = pd.read_csv(pred_csv_path)
                pred_map = dict(zip(df_pred['filename'], df_pred['predicted_eq_passage']))

            data_dir = os.path.join(root_dir, 'data', data_version, 'test')
            for class_name, def_pas in [('young', 8.0), ('senescent', 28.0)]:
                cdir = os.path.join(data_dir, class_name)
                if not os.path.exists(cdir):
                    continue
                for fname in sorted(os.listdir(cdir)):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        abs_p = os.path.join(cdir, fname)
                        pas = float(pred_map.get(fname, def_pas))
                        self.samples.append((abs_p, fname, pas))

        self.passages = np.array([s[2] for s in self.samples], dtype=np.float32)
        print(f"[MSCPassageDataset] {split}: {len(self.samples)} images | passage {self.passages.min():.1f} - {self.passages.max():.1f} (mean {self.passages.mean():.1f})")

    def __len__(self):
        return len(self.samples)

    def make_sampler(self, seed=2026, power=0.7, bin_size=1.0):
        """Passage-balanced sampler using fine 1.0-passage bins and power=0.7 to stabilize intermediate passages."""
        binned = np.round(self.passages / bin_size) * bin_size
        uniq, inv, counts = np.unique(binned, return_inverse=True, return_counts=True)
        w = 1.0 / np.power(counts[inv].astype(np.float64), power)
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
