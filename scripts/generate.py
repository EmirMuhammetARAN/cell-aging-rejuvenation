import torch
import os
import sys 
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from src.models.cyclegan_gan_model import CycleGANModel
from src.models.discriminator import Discriminator
from src.models.generator_resnet import GeneratorResNet
from torchvision import transforms
import matplotlib.pyplot as plt



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE)
model.to(DEVICE)
model.load_state_dict(torch.load('checkpoints/cyclegan_epoch5.pth', map_location=DEVICE))


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


val_img = Image.open('data/processed/val/young/senescent_MSCs_1_11.jpg').convert('RGB')
input_tensor = transform(val_img).unsqueeze(0).to(DEVICE)

model.eval()
with torch.no_grad():
    fake_senescent = model.G_AB(input_tensor)

fake_img = fake_senescent.squeeze(0).cpu()
fake_img = fake_img * 0.5 + 0.5
fake_img = fake_img.permute(1, 2, 0).numpy()

fig,axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(val_img)
axes[0].set_title('Input Young MSCs')
axes[0].axis('off')
axes[1].imshow(fake_img)
axes[1].set_title('Generated Senescent MSCs')
axes[1].axis('off')

os.makedirs(os.path.join(parent_dir, 'results'), exist_ok=True)
plt.savefig(os.path.join(parent_dir, 'results', 'generated_senescent.png'), dpi=150, bbox_inches='tight')
    