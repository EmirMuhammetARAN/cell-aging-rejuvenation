import torch
import os
import sys
from PIL import Image
import torchvision.transforms as transforms
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.models.cyclegan_gan_model import CycleGANModel
from src.models.generator_resnet import GeneratorResNet
from src.models.discriminator import Discriminator



MODEL_NAME = 'v2_epoch_50'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = os.path.join(parent_dir, 'checkpoints', f'cyclegan_{MODEL_NAME}.pth')
TEST_DIR = os.path.join(parent_dir, 'data', 'processed', 'test', 'young')
OUTPUT_DIR = os.path.join(parent_dir, 'results', f'generated_{MODEL_NAME}')
os.makedirs(OUTPUT_DIR, exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE)
model.eval()


for img_name in os.listdir(TEST_DIR):
    img = Image.open(os.path.join(TEST_DIR, img_name)).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        fake = model.G_AB(tensor)
    
    fake_img = fake.squeeze(0).cpu() * 0.5 + 0.5
    fake_img = fake_img.clamp(0, 1)
    save_img = transforms.ToPILImage()(fake_img)
    save_img.save(os.path.join(OUTPUT_DIR, img_name))
    
print(f'{len(os.listdir(OUTPUT_DIR))} resim üretildi → results/generated_{MODEL_NAME}/')