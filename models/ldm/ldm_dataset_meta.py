"""
LDM Dataset with Fill-Ratio Metadata Conditioning.
Computes cell occupancy ratio for each image and creates 
multi-class labels: class * NUM_BINS + size_bin
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision.transforms import functional as TF

NUM_BINS = 4  # small, medium, large, xlarge

def compute_fill_ratio(img_path):
    """Compute cell-to-background ratio using Otsu thresholding."""
    img = Image.open(img_path).convert('L')
    arr = np.array(img, dtype=np.float32)
    
    # Otsu threshold
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(arr)
    except Exception:
        return 0.5
    
    # Cell pixels are typically darker than background in bright-field
    cell_mask = arr < thresh
    fill_ratio = cell_mask.sum() / cell_mask.size
    return fill_ratio

def get_size_bin(fill_ratio):
    """Quantize fill ratio into bins."""
    if fill_ratio < 0.15:
        return 0  # small
    elif fill_ratio < 0.30:
        return 1  # medium
    elif fill_ratio < 0.50:
        return 2  # large
    else:
        return 3  # xlarge

class LDMDatasetMeta(Dataset):
    def __init__(self, root_dir, split='train', transform=None, data_version='processed_v2'):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []  # combined label: class * NUM_BINS + size_bin
        self.class_labels = []  # original class (0=young, 1=senescent)
        self.bin_labels = []  # size bin
        
        data_dir = os.path.join(root_dir, 'data', data_version)
        
        bin_counts = {i: 0 for i in range(NUM_BINS)}
        
        for class_name, class_id in [('young', 0), ('senescent', 1)]:
            class_dir = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    fpath = os.path.join(class_dir, fname)
                    fill_ratio = compute_fill_ratio(fpath)
                    size_bin = get_size_bin(fill_ratio)
                    combined_label = class_id * NUM_BINS + size_bin
                    
                    self.images.append(fpath)
                    self.labels.append(combined_label)
                    self.class_labels.append(class_id)
                    self.bin_labels.append(size_bin)
                    bin_counts[size_bin] = bin_counts.get(size_bin, 0) + 1
        
        num_classes = 2 * NUM_BINS  # 8
        print(f"[LDMDatasetMeta] {split}: {len(self.images)} images")
        print(f"  young={self.class_labels.count(0)}, senescent={self.class_labels.count(1)}")
        print(f"  size bins: {bin_counts}")
        print(f"  num_classes={num_classes} (2 classes x {NUM_BINS} bins)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label
