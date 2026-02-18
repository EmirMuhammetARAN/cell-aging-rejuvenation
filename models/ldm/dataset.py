from torch.utils.data import Dataset
import os
from PIL import Image
import torch
from torchvision import transforms


class LDMDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.young_images = [os.path.join(root_dir, 'young', f) for f in os.listdir(os.path.join(root_dir, 'young')) if f.endswith('.jpg')]
        self.old_images = [os.path.join(root_dir, 'senescent', f) for f in os.listdir(os.path.join(root_dir, 'senescent')) if f.endswith('.jpg')]
        self.images = self.young_images + self.old_images
        self.labels = [0] * len(self.young_images) + [1] * len(self.old_images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index])

        return image, label
    def __len__(self):
        return len(self.images)