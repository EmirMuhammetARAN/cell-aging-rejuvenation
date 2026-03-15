import os
import sys
from torchvision import transforms
import torch
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier
from PIL import Image

model = Classifier(output_size=2)
model.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints/classifier/classifier.pth')))
model.to('cuda')
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

real_young = os.path.join(root_dir, 'data', 'processed_v6', 'test', 'young')
real_senescent = os.path.join(root_dir, 'data', 'processed_v6', 'test', 'senescent')

tasks = [
    ('aging', 0, real_senescent),              
    ('rejuvenation', 1, real_young),               
    ('random_samples_young', 1, real_young),     
    ('random_samples_senescent', 0, real_senescent),  
]

ldm_test_dir = os.path.join(root_dir, 'results', 'generated', 'ldm_diffusers_v6_tightcrop')

print("=" * 60)
print("CLASSIFIER ACCURACY")
print("=" * 60)

for folder, expected_class, _ in tasks:
    folder_path = os.path.join(ldm_test_dir, folder)
    if not os.path.exists(folder_path):
        print(f'{folder}: KLASÖR YOK')
        continue
    
    correct = 0
    total = 0
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        with torch.no_grad():
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to('cuda')
            prediction = model(img)
            total += 1
            if prediction.argmax(dim=1).item() == expected_class:
                correct += 1
    
    acc = correct / total if total > 0 else 0
    print(f'{folder}: {correct}/{total} = {acc*100:.2f}%')

del model
torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("FID SCORES")
print("=" * 60)

for folder, _, real_dir in tasks:
    folder_path = os.path.join(ldm_test_dir, folder)
    if not os.path.exists(folder_path):
        print(f'{folder}: KLASÖR YOK')
        continue
    
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, folder_path],
        batch_size=32,
        device='cuda',
        dims=2048,
        num_workers=0
    )
    print(f'{folder}: FID = {fid:.2f}')
