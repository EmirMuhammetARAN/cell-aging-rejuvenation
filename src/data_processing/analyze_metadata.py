"""
via_project.json'dan hücresel metadata analizi.
Her hücre için: alan, doluluk oranı, aspect ratio, çevre uzunluğu hesaplar.
Hocanın istediği "boşluk ve doluluk" analizini yapar.
"""
import os, sys, json
import numpy as np
from PIL import Image
from skimage.draw import polygon as sk_polygon

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

# via_project.json paths (train + test + val)
VIA_PATHS = {
    'train': os.path.join(root_dir, 'data', 'raw', 'train', 'via_project.json'),
    'test': os.path.join(root_dir, 'data', 'raw', 'test', 'via_project.json'),
    'val': os.path.join(root_dir, 'data', 'raw', 'val', 'via_project.json'),
}

def compute_cell_metadata(points_x, points_y, img_w, img_h):
    """Compute morphological metadata from polygon coordinates."""
    px = np.array(points_x, dtype=np.float64)
    py = np.array(points_y, dtype=np.float64)
    
    # Bounding box
    x_min, x_max = px.min(), px.max()
    y_min, y_max = py.min(), py.max()
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    
    # Aspect ratio (width / height)
    aspect_ratio = bbox_w / max(bbox_h, 1)
    
    # Cell area (polygon area using Shoelace formula)
    n = len(px)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += px[i] * py[j]
        area -= px[j] * py[i]
    cell_area = abs(area) / 2.0
    
    # Fill ratio (cell area / image area)
    img_area = img_w * img_h
    fill_ratio = cell_area / img_area
    
    # Perimeter
    perimeter = 0.0
    for i in range(n):
        j = (i + 1) % n
        perimeter += np.sqrt((px[j] - px[i])**2 + (py[j] - py[i])**2)
    
    # Circularity (4π × area / perimeter²)  1.0 = perfect circle
    circularity = (4 * np.pi * cell_area) / max(perimeter**2, 1)
    
    return {
        'area': cell_area,
        'fill_ratio': fill_ratio,
        'aspect_ratio': aspect_ratio,
        'perimeter': perimeter,
        'circularity': circularity,
        'bbox_w': bbox_w,
        'bbox_h': bbox_h,
    }

# ============================================
# Parse all JSON files
# ============================================
young_cells = []
senescent_cells = []

for split_name, json_path in VIA_PATHS.items():
    if not os.path.exists(json_path):
        print(f"  {split_name}: via_project.json not found, skipping")
        continue
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    for key, entry in data.items():
        filename = entry.get('filename', '')
        
        # Get image dimensions
        img_dir = os.path.dirname(json_path)
        img_path = os.path.join(img_dir, filename)
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img_w, img_h = img.size
        else:
            img_w, img_h = 2592, 1944  # default microscope resolution
        
        for region in entry.get('regions', []):
            shape = region.get('shape_attributes', {})
            attrs = region.get('region_attributes', {})
            cell_class = attrs.get('cell', 'unknown')
            
            if cell_class not in ('young', 'senescent'):
                continue
            
            px = shape.get('all_points_x', [])
            py = shape.get('all_points_y', [])
            
            if len(px) < 3:
                continue
            
            meta = compute_cell_metadata(px, py, img_w, img_h)
            meta['class'] = cell_class
            meta['filename'] = filename
            meta['split'] = split_name
            
            if cell_class == 'young':
                young_cells.append(meta)
            else:
                senescent_cells.append(meta)

print(f"Parsed: {len(young_cells)} young cells, {len(senescent_cells)} senescent cells")

# ============================================
# Statistical Analysis
# ============================================
def stats(values):
    arr = np.array(values)
    return f"{arr.mean():.4f} ± {arr.std():.4f} (min={arr.min():.4f}, max={arr.max():.4f})"

print(f"\n{'='*80}")
print("MORPHOLOGICAL METADATA ANALYSIS")
print(f"{'='*80}")

metrics = ['area', 'fill_ratio', 'aspect_ratio', 'perimeter', 'circularity']
labels = {
    'area': 'Cell Area (px2)',
    'fill_ratio': 'Fill Ratio (%)',
    'aspect_ratio': 'Aspect Ratio (W/H)',
    'perimeter': 'Perimeter (px)',
    'circularity': 'Circularity (1=circle)',
}

print(f"\n{'Metrik':<30} | {'Young':<35} | {'Senescent':<35}")
print(f"{'-'*100}")

for m in metrics:
    y_vals = [c[m] for c in young_cells]
    s_vals = [c[m] for c in senescent_cells]
    
    y_mean = np.mean(y_vals)
    s_mean = np.mean(s_vals)
    
    if m == 'fill_ratio':
        # Show as percentage
        y_str = f"{np.mean(y_vals)*100:.2f}% ± {np.std(y_vals)*100:.2f}%"
        s_str = f"{np.mean(s_vals)*100:.2f}% ± {np.std(s_vals)*100:.2f}%"
    else:
        y_str = f"{np.mean(y_vals):.2f} ± {np.std(y_vals):.2f}"
        s_str = f"{np.mean(s_vals):.2f} ± {np.std(s_vals):.2f}"
    
    change = ((s_mean - y_mean) / max(y_mean, 0.001)) * 100
    print(f"{labels[m]:<30} | {y_str:<35} | {s_str:<35} | Delta={change:+.1f}%")

print(f"\n{'='*80}")
print("INTERPRETATION:")
print("  Fill Ratio: Senescent > Young -> senescent cells are wider (flattened)")
print("  Aspect Ratio: Young -> elongated (spindle), Senescent -> wide (flat)")
print("  Circularity: Senescent -> more circular (flat), Young -> elongated")
print(f"{'='*80}")
