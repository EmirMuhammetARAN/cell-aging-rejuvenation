import os, sys
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import gc

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.ldm.model_lpips import CellLDM

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ===== CONFIG =====
CHECKPOINT = os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v11_lpips_phaseB.pt')
TEST_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
TEST_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', 'v11_phaseB_full')

os.makedirs(os.path.join(OUTPUT_DIR, 'aging'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejuvenation'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'random_samples_young'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'random_samples_senescent'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load model
vae_path = os.path.join(root_dir, 'checkpoints', 'vae_finetuned', 'best')
model = CellLDM(num_classes=2, vae_path=vae_path, lpips_weight=0.0)
model.to('cuda', memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)
model.init_ema()

ckpt = torch.load(CHECKPOINT, map_location='cpu')
model.unet.load_state_dict(ckpt['unet_state_dict'])
if 'ema_unet_state_dict' in ckpt:
    model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
print(f"Loaded: epoch {ckpt.get('epoch', '?')}, val_loss {ckpt.get('val_loss', '?')}")
del ckpt; gc.collect()
model.eval()

# 1) Aging: young -> senescent
young_files = sorted(os.listdir(TEST_YOUNG_DIR))
print(f"\n[1/4] Aging translation: {len(young_files)} young -> senescent")
for img_name in tqdm(young_files, desc="Aging"):
    img_path = os.path.join(TEST_YOUNG_DIR, img_name)
    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
    with torch.no_grad():
        out = model.translate(image, target_labels=torch.tensor([1], device='cuda'),
                              strength=0.7, num_steps=50, use_ema=True, guidance_scale=4.0)
    save_image(out.cpu(), os.path.join(OUTPUT_DIR, 'aging', img_name))
    torch.cuda.empty_cache()

# 2) Rejuvenation: senescent -> young
senes_files = sorted(os.listdir(TEST_SENESCENT_DIR))
print(f"\n[2/4] Rejuvenation: {len(senes_files)} senescent -> young")
for img_name in tqdm(senes_files, desc="Rejuvenation"):
    img_path = os.path.join(TEST_SENESCENT_DIR, img_name)
    image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
    with torch.no_grad():
        out = model.translate(image, target_labels=torch.tensor([0], device='cuda'),
                              strength=0.65, num_steps=50, use_ema=True, guidance_scale=3.0)
    save_image(out.cpu(), os.path.join(OUTPUT_DIR, 'rejuvenation', img_name))
    torch.cuda.empty_cache()

# 3) Random young samples (same count as test young)
print(f"\n[3/4] Random young generation: {len(young_files)} samples")
for i in tqdm(range(len(young_files)), desc="Random Young"):
    label = torch.zeros((1,), dtype=torch.long, device='cuda')
    with torch.no_grad():
        out = model.sample(num_samples=1, device='cuda', labels=label,
                           use_ema=True, guidance_scale=3.0, num_steps=75)
    save_image(out.cpu(), os.path.join(OUTPUT_DIR, 'random_samples_young', f'young_random_sample_{i}.png'))
    torch.cuda.empty_cache()

# 4) Random senescent samples (same count as test senescent)
print(f"\n[4/4] Random senescent generation: {len(senes_files)} samples")
for i in tqdm(range(len(senes_files)), desc="Random Senescent"):
    label = torch.ones((1,), dtype=torch.long, device='cuda')
    with torch.no_grad():
        out = model.sample(num_samples=1, device='cuda', labels=label,
                           use_ema=True, guidance_scale=3.0, num_steps=75)
    save_image(out.cpu(), os.path.join(OUTPUT_DIR, 'random_samples_senescent', f'senescent_random_sample_{i}.png'))
    torch.cuda.empty_cache()

print(f"\nDone! All outputs in: {OUTPUT_DIR}")