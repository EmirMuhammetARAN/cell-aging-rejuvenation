import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import random

class LDMDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.use_augmentation = (split == 'train')
        
        self.images = []
        self.labels = [] 
        
        data_dir = os.path.join(root_dir, 'data', 'processed_v2')
        
        for class_name, label in [('young', 0), ('senescent', 1)]:
            class_dir = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    self.images.append(os.path.join(class_dir, fname))
                    self.labels.append(label)
        
        print(f"[LDMDataset] {split}: {len(self.images)} images "
              f"(young={self.labels.count(0)}, senescent={self.labels.count(1)})")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        
        if self.use_augmentation:
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            rot = random.choice([0, 1, 2, 3])
            if rot > 0:
                img = img.rotate(rot * 90, expand=False)
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label