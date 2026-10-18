"""
Thesis Figure Generator - 4-Column Benchmark Comparison:
Input | CycleGAN (Ep 160, Seed 2026) | CUT GAN (Ep 200, Seed 2026) | LDM v12 (Seed 2026, Low-FID)
Generates:
  1. figure1_aging_v2.png (Young -> Senescent)
  2. figure2_rejuvenation_v2.png (Senescent -> Young)
"""
import os, sys, random, shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ARTIFACT_DIR = r'C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7'

# Paths
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')

CYCLEGAN_AGING = os.path.join(root_dir, 'results', 'generated', 'cyclegan_seed_sweep', 'seed_2026', 'aging')
CYCLEGAN_REJU = os.path.join(root_dir, 'results', 'generated', 'cyclegan_seed_sweep', 'seed_2026', 'rejuv')

CUT_AGING = os.path.join(root_dir, 'results', 'generated', 'cut_seed_sweep', 'seed_2026', 'aging')
CUT_REJU = os.path.join(root_dir, 'results', 'generated', 'cut_seed_sweep', 'seed_2026', 'rejuv')

LDM_AGING = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026', 'aging')
LDM_REJU = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026', 'rejuv')

OUTPUT_DIR = os.path.join(root_dir, 'results', 'thesis_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_SIZE = 256
HEADER_H = 34
NUM_SAMPLES = 3

COLORS = {
    'input': (100, 110, 120),
    'cyclegan': (31, 119, 180),
    'cut': (44, 160, 44),
    'ldm': (214, 39, 40),
}


def get_font(size=14, bold=False):
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def create_figure(task, input_dir, cyclegan_dir, cut_dir, ldm_dir, title, output_name):
    # Find common images
    input_imgs = set(os.listdir(input_dir))
    cyclegan_imgs = set(os.listdir(cyclegan_dir))
    cut_imgs = set(os.listdir(cut_dir))
    ldm_imgs = set(os.listdir(ldm_dir))
    common = sorted(input_imgs & cyclegan_imgs & cut_imgs & ldm_imgs)
    
    if len(common) < NUM_SAMPLES:
        print(f"WARNING: Only {len(common)} common images found!")
        samples = common[:NUM_SAMPLES]
    else:
        random.seed(42)
        samples = random.sample(common, NUM_SAMPLES)
    
    # Canvas
    num_cols = 4
    width = CELL_SIZE * num_cols
    height = (CELL_SIZE + HEADER_H) * NUM_SAMPLES + 46  # +46 for title
    canvas = Image.new('RGB', (width, height), '#fcfcfc')
    draw = ImageDraw.Draw(canvas)
    font = get_font(13, bold=True)
    title_font = get_font(17, bold=True)
    
    # Title
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 10), title, fill='#111111', font=title_font)
    
    labels = [
        (f"Input ({task})", 'input'),
        ("CycleGAN (Ep 160)", 'cyclegan'),
        ("CUT GAN (Ep 200)", 'cut'),
        ("LDM v12 (Seed 2026)", 'ldm'),
    ]
    
    dirs = [input_dir, cyclegan_dir, cut_dir, ldm_dir]
    
    for row_idx, img_name in enumerate(samples):
        y_offset = 46 + row_idx * (CELL_SIZE + HEADER_H)
        
        for col_idx, (label, color_key) in enumerate(labels):
            x_offset = col_idx * CELL_SIZE
            
            # Header bar
            color = COLORS[color_key]
            draw.rectangle([x_offset, y_offset, x_offset + CELL_SIZE, y_offset + HEADER_H], fill=color)
            bbox = draw.textbbox((0, 0), label, font=font)
            lw = bbox[2] - bbox[0]
            draw.text((x_offset + (CELL_SIZE - lw) // 2, y_offset + 8), label, fill='white', font=font)
            
            # Image
            img_path = os.path.join(dirs[col_idx], img_name)
            img = Image.open(img_path).convert('RGB').resize((CELL_SIZE, CELL_SIZE), Image.Resampling.LANCZOS)
            canvas.paste(img, (x_offset, y_offset + HEADER_H))
            
            # Subtle border around image
            draw.rectangle([x_offset, y_offset + HEADER_H, x_offset + CELL_SIZE, y_offset + HEADER_H + CELL_SIZE], outline='#dddddd', width=1)
    
    out_path = os.path.join(OUTPUT_DIR, output_name)
    canvas.save(out_path, quality=95)
    print(f"  [OK] Saved: {out_path}")
    
    # Copy to artifact directory if available
    if os.path.exists(ARTIFACT_DIR):
        art_out = os.path.join(ARTIFACT_DIR, output_name)
        shutil.copy2(out_path, art_out)
        print(f"  [OK] Copied to artifact: {art_out}")
        
    return out_path


if __name__ == "__main__":
    print("Creating updated 4-column thesis figures (CycleGAN 160 vs CUT 200 vs LDM v12 Seed 2026)...\n")
    
    # Figure 1: Aging (Young -> Senescent)
    create_figure(
        "Young", TEST_YOUNG, CYCLEGAN_AGING, CUT_AGING, LDM_AGING,
        "AGING TASK (Young -> Senescent)", "figure1_aging_v2.png"
    )
    
    # Figure 2: Rejuvenation (Senescent -> Young)
    create_figure(
        "Senescent", TEST_SENES, CYCLEGAN_REJU, CUT_REJU, LDM_REJU,
        "REJUVENATION TASK (Senescent -> Young)", "figure2_rejuvenation_v2.png"
    )
    
    print("\n[DONE] Figures created successfully!")

