import os
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class DatasetMaker(Dataset):
    def __init__(self, young_path, senescent_path, transform=None):
        self.young_path = young_path
        self.senescent_path = senescent_path
        self.young_files = os.listdir(os.path.join(self.young_path))
        self.senescent_files = os.listdir(os.path.join(self.senescent_path))
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
    