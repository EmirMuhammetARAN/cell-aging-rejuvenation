import os, sys
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.ldm_dataset import LDMDataset
from diffusers import AutoencoderKL

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

vae_path = os.path.join(root_dir, 'checkpoints', 'vae_finetuned', 'best')
print(f"VAE loading: {vae_path}")
vae = AutoencoderKL.from_pretrained(vae_path).to(DEVICE)
vae.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
dataset = LDMDataset(root_dir=root_dir, split='train', transform=transform, data_version='processed_v2')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print("Collecting latent matrices...")
all_latents = []

with torch.no_grad():
    for i, (images, _) in enumerate(dataloader):
        if i >= 10:
            break
        images = images.to(DEVICE)
        latent_dist = vae.encode(images).latent_dist
        latents = latent_dist.sample()
        all_latents.append(latents.cpu())

all_latents = torch.cat(all_latents, dim=0)

std = all_latents.std().item()
mean = all_latents.mean().item()

print("\n" + "="*50)
print(f"Sample Count: {len(all_latents)} cells")
print(f"Latent Mean: {mean:.4f}")
print(f"Latent Std: {std:.4f}")
print("-" * 50)
print(f"SCALING FACTOR (1 / Std) = {1.0 / std:.5f}")
print("="*50)