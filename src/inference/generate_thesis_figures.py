"""
Generate thesis comparison figures:
  Figure 1: AGING   - Input(Young) | CycleGAN | LDM  (3 rows)
  Figure 2: REJUVENATION - Input(Senescent) | CycleGAN | LDM  (3 rows)
"""
import os, sys, torch, gc, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.utils import save_image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, root_dir)

from models.cyclegan.generator_resnet import GeneratorResNet
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.cyclegan_gan_model import CycleGANModel
from models.ldm.model_lpips import CellLDM

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = 'cuda'

# --- Paths ---
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')
CYCLEGAN_CKPT = os.path.join(root_dir, 'checkpoints', 'cyclegan_v2_epoch_110.pth')
LDM_CKPT = os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v2_lpips_ft.pt')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'thesis_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Config ---
N_SAMPLES = 3  # rows per figure
IMG_SIZE = 256  # display size per cell
SEED = 42
random.seed(SEED)

# LDM optimal params
AGING_STRENGTH = 0.8
AGING_CFG = 5.0
REJU_STRENGTH = 0.7
REJU_CFG = 4.0

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def tensor_to_pil(t):
    """Convert [-1,1] or [0,1] tensor to PIL."""
    t = t.squeeze(0).cpu()
    if t.min() < 0:
        t = t * 0.5 + 0.5
    t = t.clamp(0, 1)
    return transforms.ToPILImage()(t)

def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()

def add_label(img, text, color):
    """Add colored label bar on top of image."""
    draw = ImageDraw.Draw(img)
    bar_h = 28
    draw.rectangle([(0, 0), (img.width, bar_h)], fill=color)
    font = get_font(16)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.width - tw) // 2, 5), text, fill='white', font=font)
    return img

def add_title(grid, title, grid_w):
    """Add a title bar at the top of the grid."""
    title_h = 40
    new_img = Image.new('RGB', (grid_w, grid.height + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(new_img)
    font = get_font(22)
    bbox = draw.textbbox((0, 0), title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((grid_w - tw) // 2, 8), title, fill='black', font=font)
    new_img.paste(grid, (0, title_h))
    return new_img

# =============================================
# 1. Load CycleGAN v2 (epoch 110)
# =============================================
print("Loading CycleGAN v2 (epoch 110)...")
cyclegan = CycleGANModel(GeneratorResNet, Discriminator, DEVICE, use_lpips=False)
cyclegan.load_state_dict(torch.load(CYCLEGAN_CKPT, map_location=DEVICE))
cyclegan.to(DEVICE, memory_format=torch.channels_last)
cyclegan.eval()

# =============================================
# 2. Load LDM (optimized)
# =============================================
print("Loading LDM (optimized)...")
ldm = CellLDM(num_classes=2, lpips_weight=0.0)
ldm.to(DEVICE, memory_format=torch.channels_last)
ldm.vae.to(memory_format=torch.channels_last)
ldm.init_ema()
ckpt = torch.load(LDM_CKPT, map_location='cpu')
ldm.unet.load_state_dict(ckpt['unet_state_dict'])
if 'ema_unet_state_dict' in ckpt:
    ldm.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
del ckpt; gc.collect()
ldm.eval()

# =============================================
# Pick random samples
# =============================================
young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.startswith('young_')])
senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.startswith('senescent_')])

# Filter for variety
young_samples = random.sample(young_files, min(N_SAMPLES, len(young_files)))
senes_samples = random.sample(senes_files, min(N_SAMPLES, len(senes_files)))

print(f"Young samples: {young_samples}")
print(f"Senescent samples: {senes_samples}")

# =============================================
# FIGURE 1: AGING (Young -> Senescent)
# =============================================
print("\n--- Generating Figure 1: AGING ---")
rows = []
for img_name in young_samples:
    img_path = os.path.join(TEST_YOUNG, img_name)
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    
    # Original
    orig = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    orig = add_label(orig, "Input (Young)", "#4a4a4a")
    
    # CycleGAN
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cyc_out = cyclegan.G_AB(img_tensor)
    cyc_pil = tensor_to_pil(cyc_out).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    cyc_pil = add_label(cyc_pil, "CycleGAN", "#4285f4")
    
    # LDM
    with torch.no_grad():
        ldm_out = ldm.translate(img_tensor, target_labels=torch.tensor([1], device=DEVICE),
                                strength=AGING_STRENGTH, num_steps=50,
                                use_ema=True, guidance_scale=AGING_CFG)
    ldm_pil = tensor_to_pil(ldm_out).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    ldm_pil = add_label(ldm_pil, "LDM", "#34a853")
    
    rows.append((orig, cyc_pil, ldm_pil))
    torch.cuda.empty_cache()

# Stitch into grid
grid_w = IMG_SIZE * 3 + 4  # 2px gap between columns
grid_h = IMG_SIZE * N_SAMPLES + (N_SAMPLES - 1) * 2
grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
for i, (a, b, c) in enumerate(rows):
    y = i * (IMG_SIZE + 2)
    grid.paste(a, (0, y))
    grid.paste(b, (IMG_SIZE + 2, y))
    grid.paste(c, (IMG_SIZE * 2 + 4, y))

grid = add_title(grid, 'AGING (Young \u2192 Senescent)', grid_w)
grid.save(os.path.join(OUTPUT_DIR, 'figure1_aging.png'), quality=95)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'figure1_aging.png')}")

# =============================================
# FIGURE 2: REJUVENATION (Senescent -> Young)
# =============================================
print("\n--- Generating Figure 2: REJUVENATION ---")
rows = []
for img_name in senes_samples:
    img_path = os.path.join(TEST_SENES, img_name)
    img_pil = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    
    # Original
    orig = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    orig = add_label(orig, "Input (Senescent)", "#4a4a4a")
    
    # CycleGAN
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            cyc_out = cyclegan.G_BA(img_tensor)
    cyc_pil = tensor_to_pil(cyc_out).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    cyc_pil = add_label(cyc_pil, "CycleGAN", "#4285f4")
    
    # LDM
    with torch.no_grad():
        ldm_out = ldm.translate(img_tensor, target_labels=torch.tensor([0], device=DEVICE),
                                strength=REJU_STRENGTH, num_steps=50,
                                use_ema=True, guidance_scale=REJU_CFG)
    ldm_pil = tensor_to_pil(ldm_out).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    ldm_pil = add_label(ldm_pil, "LDM", "#34a853")
    
    rows.append((orig, cyc_pil, ldm_pil))
    torch.cuda.empty_cache()

# Stitch
grid = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
for i, (a, b, c) in enumerate(rows):
    y = i * (IMG_SIZE + 2)
    grid.paste(a, (0, y))
    grid.paste(b, (IMG_SIZE + 2, y))
    grid.paste(c, (IMG_SIZE * 2 + 4, y))

grid = add_title(grid, 'REJUVENATION (Senescent \u2192 Young)', grid_w)
grid.save(os.path.join(OUTPUT_DIR, 'figure2_rejuvenation.png'), quality=95)
print(f"Saved: {os.path.join(OUTPUT_DIR, 'figure2_rejuvenation.png')}")

# Cleanup
del cyclegan, ldm
torch.cuda.empty_cache(); gc.collect()
print("\n✅ Done! Figures saved to:", OUTPUT_DIR)
