import os
import sys
import numpy as np
from PIL import Image
import tifffile
import torch
from torchvision import transforms
import torch.nn.functional as F

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname((os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.classifier.classifier import Classifier

UNLABELED_DATA_PATH = "data/unlabeled_data"
VAR_THRESHOLD = 50

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = Classifier(output_size=2)
model.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier.pth')))
model.eval()
model.to('cuda')

for filename in os.listdir(UNLABELED_DATA_PATH):
    image = tifffile.imread(os.path.join(UNLABELED_DATA_PATH, filename))
    h, w, _ = image.shape
    y_positions = [0, 512, 1024, 1432]       
    x_positions = [0, 512, 1024, 1536, 2048]  

    for y in y_positions:
        for x in x_positions:
            patch = image[y:y+512, x:x+512]
            
            variance = np.var(patch)
            base = os.path.splitext(filename)[0]

            if variance < VAR_THRESHOLD:
                patch_filename = f"{base}_{y}_{x}_var{variance:.0f}.tif"
                patch_path = os.path.join("data/labeled_data/rejected", patch_filename)
                os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                tifffile.imwrite(patch_path, patch)
            else:
                patchImage = Image.fromarray(patch)
                transformed_patch = transform(patchImage).unsqueeze(0).to('cuda')
                with torch.no_grad():
                    output = model(transformed_patch)
                    _, predicted = torch.max(output, 1)
                    label_name = "young" if predicted.item() == 0 else "senescent"
                patch_filename = f"{base}_{y}_{x}.jpg"
                patch_path = os.path.join(f"data/labeled_data/{label_name}", patch_filename)
                os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                Image.fromarray(patch).save(patch_path)

young_count = len(os.listdir("data/labeled_data/young")) if os.path.exists("data/labeled_data/young") else 0
senes_count = len(os.listdir("data/labeled_data/senescent")) if os.path.exists("data/labeled_data/senescent") else 0
rejected_count = len(os.listdir("data/labeled_data/rejected")) if os.path.exists("data/labeled_data/rejected") else 0
print(f"Young: {young_count}, Senescent: {senes_count}, Rejected: {rejected_count}")