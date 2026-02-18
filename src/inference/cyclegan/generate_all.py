import torch
import os
import sys
from PIL import Image
import torchvision.transforms as transforms

# Get root directory
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.cyclegan.cyclegan_gan_model import CycleGANModel
from models.cyclegan.generator_resnet import GeneratorResNet
from models.cyclegan.discriminator import Discriminator



MODEL_NAME = 'v1_epoch_70' 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = os.path.join(root_dir, 'checkpoints', f'cyclegan_{MODEL_NAME}.pth')
TEST_DIR = os.path.join(root_dir, 'data', 'processed', 'test', 'young')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', MODEL_NAME,'aging')
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
    
print(f'{len(os.listdir(OUTPUT_DIR))} resim üretildi → results/generated/{MODEL_NAME}/')