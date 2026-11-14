import os
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class DatasetMaker(Dataset):
    """Unpaired young vs senescent image dataset with deterministic sorted file order."""

    VALID_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')

    def __init__(self, young_path, senescent_path, transform=None):
        self.young_path = young_path
        self.senescent_path = senescent_path
        self.young_files = sorted(
            f for f in os.listdir(self.young_path) if f.lower().endswith(self.VALID_EXTS)
        )
        self.senescent_files = sorted(
            f for f in os.listdir(self.senescent_path) if f.lower().endswith(self.VALID_EXTS)
        )
        self.transform = transform
    
    def __len__(self):
        return max(len(self.young_files), len(self.senescent_files))
    
    def __getitem__(self, idx):
        young_file = self.young_files[idx % len(self.young_files)]
        senescent_file = self.senescent_files[random.randint(0, len(self.senescent_files) - 1)]

        young_img = Image.open(os.path.join(self.young_path, young_file)).convert('RGB')
        senescent_img = Image.open(os.path.join(self.senescent_path, senescent_file)).convert('RGB')

        if self.transform:
            young_img = self.transform(young_img)
            senescent_img = self.transform(senescent_img)
        
        return {"A": young_img, "B":senescent_img}
    