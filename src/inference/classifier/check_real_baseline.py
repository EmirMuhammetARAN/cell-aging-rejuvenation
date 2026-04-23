import os, sys
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier
from torchvision.datasets import ImageFolder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load classifier
classifier = Classifier(output_size=2).to(DEVICE)
classifier_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
classifier.eval()

# Check ImageFolder mapping
dummy_dataset = ImageFolder(root=os.path.join(root_dir, 'data', 'processed', 'train'))
print(f"Classifier Class-to-Idx Mapping: {dummy_dataset.class_to_idx}")
# Likely {senescent: 0, young: 1}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

REAL_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
REAL_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')

def evaluate_dir(directory, target_label):
    correct = 0
    total = 0
    images = os.listdir(directory)
    for img_name in tqdm(images):
        img_path = os.path.join(directory, img_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = classifier(image)
            pred = torch.argmax(output, dim=1).item()
            if pred == target_label:
                correct += 1
            total += 1
    return correct, total

print("Evaluating Classifier on REAL TEST DATA (processed_v2)")
s_corr, s_tot = evaluate_dir(REAL_SENESCENT_DIR, 0) # senescent = 0
y_corr, y_tot = evaluate_dir(REAL_YOUNG_DIR, 1) # young = 1

print(f"Real Senescent Accuracy: {s_corr}/{s_tot} = {s_corr/s_tot*100:.2f}%")
print(f"Real Young Accuracy: {y_corr}/{y_tot} = {y_corr/y_tot*100:.2f}%")
