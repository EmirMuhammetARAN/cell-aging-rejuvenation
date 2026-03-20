import os, sys, gc
from copy import deepcopy
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.model_diffusers import CellLDM

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v6', 'test', 'young')
TEST_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v6', 'test', 'senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm_diffusers_v6_tightcrop_ep200')

os.makedirs(os.path.join(OUTPUT_DIR, 'aging'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejuvenation'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'random_samples_young'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'random_samples_senescent'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# --- Load model ---
print("Loading model...")
model = CellLDM(num_classes=2)
model.to(DEVICE, memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)
model.init_ema()

checkpoint_path = 'checkpoints/ldm/v7/checkpoint_v6_tightcrop_epoch_200.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# LoRA unload (eğer varsa)
if hasattr(model.unet, 'peft_config'):
    print("Unloading existing LoRA...")
    model.unet = model.unet.unload()

# LoRA ekle (yeni)
from peft import get_peft_model
model.unet = get_peft_model(model.unet, model.loraconfig)
print("✓ LoRA applied")

# State dict yükle
if 'unet_state_dict' in checkpoint:
    model.unet.load_state_dict(checkpoint['unet_state_dict'], strict=False)
    print("✓ UNet weights loaded")

if 'class_embed_state_dict' in checkpoint:
    model.class_embed.load_state_dict(checkpoint['class_embed_state_dict'])
    print("✓ Class embedding loaded")

# EMA UNet
model.ema_unet = deepcopy(model.unet)
model.ema_unet.requires_grad_(False)
model.ema_unet.eval()

print(f"✓ Model loaded (epoch: {checkpoint.get('epoch', 'N/A')})")
del checkpoint
gc.collect()
torch.cuda.empty_cache()

model.eval()

# --- 1) Translation: Young → Senescent (aging) ---
young_images = sorted([f for f in os.listdir(TEST_YOUNG_DIR) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))])
print(f"\n--- Aging (Young → Senescent): {len(young_images)} images ---")
skipped = 0
for img_name in tqdm(young_images, desc="Aging"):
    out_name = os.path.splitext(img_name)[0] + '.png'
    out_path = os.path.join(OUTPUT_DIR, 'aging', out_name)
    if os.path.exists(out_path):
        skipped += 1
        continue
    try:
        img_path = os.path.join(TEST_YOUNG_DIR, img_name)
        image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
        with torch.no_grad():
            generated = model.translate(image, target_labels=torch.tensor([1], device=DEVICE),
                                         strength=0.7, num_steps=50, use_ema=True, guidance_scale=3.0)
        save_image(generated.cpu(), out_path)
        del image, generated
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        continue

if skipped: print(f"  ({skipped} already existed, skipped)")

# --- 2) Translation: Senescent → Young (rejuvenation) ---
senes_images = sorted([f for f in os.listdir(TEST_SENESCENT_DIR) if f.endswith(('.jpg', '.png', '.jpeg', '.tif'))])
print(f"\n--- Rejuvenation (Senescent → Young): {len(senes_images)} images ---")
skipped = 0
for img_name in tqdm(senes_images, desc="Rejuvenation"):
    out_name = os.path.splitext(img_name)[0] + '.png'
    out_path = os.path.join(OUTPUT_DIR, 'rejuvenation', out_name)
    if os.path.exists(out_path):
        skipped += 1
        continue
    try:
        img_path = os.path.join(TEST_SENESCENT_DIR, img_name)
        image = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
        with torch.no_grad():
            generated = model.translate(image, target_labels=torch.tensor([0], device=DEVICE),
                                         strength=0.7, num_steps=50, use_ema=True, guidance_scale=3.0)
        save_image(generated.cpu(), out_path)
        del image, generated
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        continue

if skipped: print(f"  ({skipped} already existed, skipped)")

# --- 3) Random generation: Young ---
num_young = len(young_images)
print(f"\n--- Random Young Samples: {num_young} ---")
skipped = 0
for i in tqdm(range(num_young), desc="Random Young"):
    out_path = os.path.join(OUTPUT_DIR, 'random_samples_young', f'young_random_{i}.png')
    if os.path.exists(out_path):
        skipped += 1
        continue
    try:
        label = torch.zeros((1,), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            generated = model.sample(num_samples=1, device=DEVICE, labels=label,
                                      use_ema=True, guidance_scale=3.0)
        save_image(generated.cpu(), out_path)
        del generated
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error generating young sample {i}: {e}")
        continue

if skipped: print(f"  ({skipped} already existed, skipped)")

# --- 4) Random generation: Senescent ---
num_senes = len(senes_images)
print(f"\n--- Random Senescent Samples: {num_senes} ---")
skipped = 0
for i in tqdm(range(num_senes), desc="Random Senescent"):
    out_path = os.path.join(OUTPUT_DIR, 'random_samples_senescent', f'senescent_random_{i}.png')
    if os.path.exists(out_path):
        skipped += 1
        continue
    try:
        label = torch.ones((1,), dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            generated = model.sample(num_samples=1, device=DEVICE, labels=label,
                                      use_ema=True, guidance_scale=3.0)
        save_image(generated.cpu(), out_path)
        del generated
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Error generating senescent sample {i}: {e}")
        continue

if skipped: print(f"  ({skipped} already existed, skipped)")

print(f"\n✅ Tamamlandı! Sonuçlar: {OUTPUT_DIR}")