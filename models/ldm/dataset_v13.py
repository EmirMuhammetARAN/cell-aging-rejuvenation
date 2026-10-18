"""
Dataset loader for LDM v13:
Loads 512x512 images along with continuous biological senescence scores [0.0, 1.0]
pre-computed via Classifier v3 (ConvNeXt-Tiny) on the Picard Lifespan Scale.
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class LDMDatasetV13(Dataset):
    def __init__(self, root_dir, split='train', transform=None, data_version='processed_v4', cache_file=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        if cache_file is None:
            # Prefer neutral masked scores if available
            p_masked = os.path.join(root_dir, 'data', data_version, 'train_continuous_ages_neutral_masked.pt')
            p_standard = os.path.join(root_dir, 'data', data_version, 'train_continuous_ages.pt')
            cache_path = p_masked if os.path.exists(p_masked) else p_standard
        else:
            cache_path = os.path.join(root_dir, 'data', data_version, cache_file)

        self.age_cache = torch.load(cache_path, map_location='cpu') if os.path.exists(cache_path) else {}
        if os.path.exists(cache_path):
            print(f"[LDMDatasetV13] Using age cache from: {cache_path} ({len(self.age_cache)} scores loaded)")

        self.samples = []
        data_dir = os.path.join(root_dir, 'data', data_version, split)
        
        for class_name, default_score in [('young', 0.25), ('senescent', 0.50)]:
            cdir = os.path.join(data_dir, class_name)
            if not os.path.exists(cdir):
                continue
            for fname in sorted(os.listdir(cdir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    abs_path = os.path.join(cdir, fname)
                    rel_path = os.path.relpath(abs_path, root_dir)
                    norm_rel = rel_path.replace('\\', '/')
                    # Look up score by fname, rel_path, or norm_rel
                    score = self.age_cache.get(fname, self.age_cache.get(rel_path, self.age_cache.get(norm_rel, default_score)))
                    self.samples.append((abs_path, float(score)))

        print(f"[LDMDatasetV13] {split}: Loaded {len(self.samples)} images with continuous biological age scores.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age_score = self.samples[idx]
        with Image.open(path) as img:
            rgb = img.convert('RGB')
        
        if self.transform:
            rgb = self.transform(rgb)

        return rgb, torch.tensor(age_score, dtype=torch.float32)
