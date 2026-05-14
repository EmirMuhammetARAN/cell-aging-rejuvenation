"""
Thesis Figure Generator - CycleGAN v2 vs LDM v4 (FID-optimized) comparison
3 ornek x 3 sutun: Input | CycleGAN | LDM
"""
import os, sys, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Paths
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CYCLEGAN_AGING = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2_sweep', 'epoch_110', 'aging')
CYCLEGAN_REJU = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2_sweep', 'epoch_110', 'rejuvenation')
LDM_AGING = os.path.join(root_dir, 'results', 'generated', 'ldm', 'sweep_as0.75_ac4.0_rs0.65_rc3.5', 'aging')
LDM_REJU = os.path.join(root_dir, 'results', 'generated', 'ldm', 'sweep_as0.75_ac4.0_rs0.65_rc3.5', 'rejuvenation')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'thesis_figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CELL_SIZE = 256
HEADER_H = 30
NUM_SAMPLES = 3
COLORS = {
    'input': (120, 120, 120),
    'cyclegan': (66, 133, 244),
    'ldm': (52, 168, 83),
}


def get_font(size=14):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except:
        return ImageFont.load_default()


def create_figure(task, input_dir, cyclegan_dir, ldm_dir, title, output_name):
    # Find common images
    input_imgs = set(os.listdir(input_dir))
    cyclegan_imgs = set(os.listdir(cyclegan_dir))
    ldm_imgs = set(os.listdir(ldm_dir))
    common = sorted(input_imgs & cyclegan_imgs & ldm_imgs)
    
    if len(common) < NUM_SAMPLES:
        print(f"WARNING: Only {len(common)} common images found!")
        samples = common[:NUM_SAMPLES]
    else:
        random.seed(42)
        samples = random.sample(common, NUM_SAMPLES)
    
    # Canvas
    width = CELL_SIZE * 3
    height = (CELL_SIZE + HEADER_H) * NUM_SAMPLES + 40  # +40 for title
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = get_font(14)
    title_font = get_font(18)
    
    # Title
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((width - tw) // 2, 8), title, fill='black', font=title_font)
    
    labels = [
        (f"Input ({task})", 'input'),
        ("CycleGAN v2", 'cyclegan'),
        ("LDM v4", 'ldm'),
    ]
    
    for row_idx, img_name in enumerate(samples):
        y_offset = 40 + row_idx * (CELL_SIZE + HEADER_H)
        
        dirs = [input_dir, cyclegan_dir, ldm_dir]
        
        for col_idx, (label, color_key) in enumerate(labels):
            x_offset = col_idx * CELL_SIZE
            
            # Header bar
            color = COLORS[color_key]
            draw.rectangle([x_offset, y_offset, x_offset + CELL_SIZE, y_offset + HEADER_H], fill=color)
            bbox = draw.textbbox((0, 0), label, font=font)
            lw = bbox[2] - bbox[0]
            draw.text((x_offset + (CELL_SIZE - lw) // 2, y_offset + 6), label, fill='white', font=font)
            
            # Image
            img_path = os.path.join(dirs[col_idx], img_name)
            img = Image.open(img_path).convert('RGB').resize((CELL_SIZE, CELL_SIZE))
            canvas.paste(img, (x_offset, y_offset + HEADER_H))
    
    out_path = os.path.join(OUTPUT_DIR, output_name)
    canvas.save(out_path, quality=95)
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    print("Creating thesis figures (CycleGAN v2 vs LDM v4 FID-optimized)...\n")
    
    # Figure 1: Aging (Young -> Senescent)
    input_label = "Young"
    create_figure(
        input_label, TEST_YOUNG, CYCLEGAN_AGING, LDM_AGING,
        "AGING (Young -> Senescent)", "figure1_aging_v2.png"
    )
    
    # Figure 2: Rejuvenation (Senescent -> Young)
    input_label = "Senescent"
    create_figure(
        input_label, TEST_SENES, CYCLEGAN_REJU, LDM_REJU,
        "REJUVENATION (Senescent -> Young)", "figure2_rejuvenation_v2.png"
    )
    
    print("\nDone!")
