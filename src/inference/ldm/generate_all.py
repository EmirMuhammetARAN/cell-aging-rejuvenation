import os, sys

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image


current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.ldm_dataset import LDMDataset
from models.ldm.model import CellLDM
from torch.utils.data import DataLoader



TEST_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v2','test','young')
TEST_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v2','test','senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm_test')
os.makedirs(os.path.join(OUTPUT_DIR, 'aging'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejuvenation'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'random_samples_young'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'random_samples_senescent'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = CellLDM(num_classes=2)
model.to('cuda', memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)
model_checkpoint = torch.load(os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model.pt'))
model.unet.load_state_dict(model_checkpoint['unet_state_dict'])
model.init_ema()
model.ema_unet.load_state_dict(model_checkpoint['ema_unet_state_dict'])
model.eval()

for img in os.listdir(TEST_YOUNG_DIR):
    img_path = os.path.join(TEST_YOUNG_DIR, img)
    image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
    with torch.no_grad():
        generated_image = model.translate(image, target_labels=torch.tensor([1], device='cuda'), strength=0.6, num_steps=100, use_ema=True)
    save_image(generated_image.cpu(), os.path.join(OUTPUT_DIR, 'aging', img))

for img in os.listdir(TEST_SENESCENT_DIR):
    img_path = os.path.join(TEST_SENESCENT_DIR, img)
    image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
    with torch.no_grad():
        generated_image = model.translate(image, target_labels=torch.tensor([0], device='cuda'), strength=0.6, num_steps=100, use_ema=True)
    save_image(generated_image.cpu(), os.path.join(OUTPUT_DIR, 'rejuvenation', img))


for i in range(329):
    label_tensor = torch.zeros((1,), dtype=torch.long, device='cuda')
    with torch.no_grad():
        generated_image = model.sample(num_samples=1, device='cuda', labels=label_tensor, use_ema=True)
    save_image(generated_image.cpu(), os.path.join(OUTPUT_DIR ,'random_samples_young',f'young_random_sample_{i}.png'))


for i in range(319):
    label_tensor = torch.ones((1,), dtype=torch.long, device='cuda')
    with torch.no_grad():
        generated_image = model.sample(num_samples=1, device='cuda', labels=label_tensor, use_ema=True)
    save_image(generated_image.cpu(), os.path.join(OUTPUT_DIR ,'random_samples_senescent',f'senescent_random_sample_{i}.png'))