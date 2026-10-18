import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

class LDMDatasetV14(Dataset):
    """
    Dataset for LDM v14 with synchronized mask loading, direction-specific dilation,
    and synchronized spatial augmentations (random horizontal/vertical flips).
    
    Optimal Dilation Radii:
      - Young (label 0): 24 px
      - Senescent (label 1): 8 px
    """
    def __init__(self, root_dir, split='train', transform=None, data_version='processed_v4',
                 young_dilation=24, senes_dilation=8):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.young_dilation = young_dilation
        self.senes_dilation = senes_dilation
        self.is_train = (split == 'train')
        
        self.images = []
        self.labels = []
        self.fnames = []
        
        data_dir = os.path.join(root_dir, 'data', data_version)
        for class_name, label in [('young', 0), ('senescent', 1)]:
            class_dir = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(label)
                    self.fnames.append(fname)
                    
        # Load precomputed Mask R-CNN masks
        scratch_dir = os.path.join(root_dir, 'scratch')
        if self.is_train:
            p_young = os.path.join(scratch_dir, 'mrcnn_masks_train_young.npy')
            p_senes = os.path.join(scratch_dir, 'mrcnn_masks_train_senescent.npy')
        else:
            p_young = os.path.join(scratch_dir, 'mrcnn_masks_input_young.npy')
            p_senes = os.path.join(scratch_dir, 'mrcnn_masks_input_senescent.npy')
            
        self.masks_young = {}
        self.masks_senes = {}
        if os.path.exists(p_young):
            raw_y = np.load(p_young, allow_pickle=True)
            self.masks_young = raw_y.item() if hasattr(raw_y, 'item') else dict(raw_y)
        if os.path.exists(p_senes):
            raw_s = np.load(p_senes, allow_pickle=True)
            self.masks_senes = raw_s.item() if hasattr(raw_s, 'item') else dict(raw_s)

        print(f"[LDMDatasetV14] {split}: {len(self.images)} images "
              f"(young={self.labels.count(0)}, senescent={self.labels.count(1)}) | "
              f"Masks: young={len(self.masks_young)}, senescent={len(self.masks_senes)}")
              
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        fname = self.fnames[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        # Retrieve raw mask (bool array [512, 512])
        mask_dict = self.masks_young if label == 0 else self.masks_senes
        raw_mask = mask_dict.get(fname, None)
        
        d = self.young_dilation if label == 0 else self.senes_dilation
        
        if raw_mask is not None and np.any(raw_mask):
            m_t = torch.from_numpy(raw_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            dilated = F.max_pool2d(m_t, kernel_size=2 * d + 1, stride=1, padding=d)
            # Downsample to latent 64x64
            mask_latent = F.interpolate(dilated, size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
        else:
            # Fallback to full foreground (no background suppression)
            mask_latent = torch.ones(1, 64, 64, dtype=torch.float32)
            
        # Synchronized Data Augmentation (Random Flips)
        if self.is_train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_latent = torch.flip(mask_latent, dims=[2])
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask_latent = torch.flip(mask_latent, dims=[1])
                
        # Transform image
        if self.transform:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)
            img = TF.normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        return img, label, mask_latent
