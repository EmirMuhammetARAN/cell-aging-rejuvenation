"""Combine GradCAM grids into single thesis figure"""
from PIL import Image, ImageDraw, ImageFont
import os

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
grid_dir = os.path.join(root, 'results', 'gradcam', 'grid')

young = Image.open(os.path.join(grid_dir, 'gradcam_grid_young.png'))
senes = Image.open(os.path.join(grid_dir, 'gradcam_grid_senescent.png'))

w, h = young.size
header = 35
gap = 10
canvas = Image.new('RGB', (w * 2 + gap, h + header), 'white')
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

# Headers
for i, (label, color) in enumerate([("Young Cells", (52, 168, 83)), ("Senescent Cells", (234, 67, 53))]):
    x = i * (w + gap)
    draw.rectangle([x, 0, x + w, header], fill=color)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((x + (w - tw) // 2, 8), label, fill='white', font=font)

canvas.paste(young, (0, header))
canvas.paste(senes, (w + gap, header))

out = os.path.join(root, 'results', 'thesis_figures', 'preprint_figure4_gradcam.png')
os.makedirs(os.path.dirname(out), exist_ok=True)
canvas.save(out, quality=95)
print(f"Saved: {out}")
