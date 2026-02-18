import os
import sys
from torchvision import transforms
import torch

# Get root directory
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier
from PIL import Image

RESULTS_PATH_NAME = 'generated/v1_epoch_70/rejuvenation'
TEST_YOUNG_PATH = os.path.join(root_dir, 'data/processed/test/young')
TEST_OLD_PATH = os.path.join(root_dir, 'data/processed/test/senescent')
RESULTS_PATH = os.path.join(root_dir, f'results/{RESULTS_PATH_NAME}')  
CORRECT_OF_YOUNG = 0
CORRECT_OF_OLD = 0
CORRECT_OF_RESULTS = 0
IMG_COUNT = 0

model = Classifier(output_size=2)
model.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints/classifier.pth')))
model.to('cuda')
model.eval()

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

for img_name in os.listdir(TEST_YOUNG_PATH):
    img_path = os.path.join(TEST_YOUNG_PATH, img_name)
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to('cuda')
        prediction = model(img)
        IMG_COUNT += 1
        if prediction.argmax(dim=1).item() == 1:
            CORRECT_OF_YOUNG += 1

print(f'Young Accuracy: {CORRECT_OF_YOUNG/IMG_COUNT:.4f}')
IMG_COUNT = 0

for img_name in os.listdir(TEST_OLD_PATH):
    img_path = os.path.join(TEST_OLD_PATH, img_name)
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to('cuda')
        prediction = model(img)
        IMG_COUNT += 1
        if prediction.argmax(dim=1).item() == 0:
            CORRECT_OF_OLD += 1

print(f'Old Accuracy: {CORRECT_OF_OLD/IMG_COUNT:.4f}')
IMG_COUNT = 0

for img_name in os.listdir(RESULTS_PATH):
    img_path = os.path.join(RESULTS_PATH, img_name)
    with torch.no_grad():
        img = Image.open(img_path).convert('RGB')
        img = transform(img).unsqueeze(0).to('cuda')
        prediction = model(img)
        IMG_COUNT += 1
        if prediction.argmax(dim=1).item() == 1:
            CORRECT_OF_RESULTS += 1

print(f'Generated Accuracy: {CORRECT_OF_RESULTS/IMG_COUNT:.4f}')
