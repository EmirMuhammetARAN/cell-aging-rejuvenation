"""
Unlabeled TIF Pipeline v2:
1. Cellpose ile hucreleri segmente et
2. Hucre merkezinden 512x512 pencere kes (dinamik zoom YOK)
3. Kenardaysa merkezi kaydır
4. Classifier ile siniflandir (%95+ guvenli)
5. data/pseudo_labeled/ klasorune kaydet
"""
import os, sys
import numpy as np
from PIL import Image
from cellpose import models
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

ROOT = r"c:\Users\emir_\Documents\GitHub\Yeni dizin\dataset_senescence"
sys.path.insert(0, ROOT)

UNLABELED_DIR = os.path.join(ROOT, 'data', 'unlabeled_data')
CELLPOSE_MODEL = os.path.join(ROOT, 'checkpoints', 'cellpose', 'models', 'cellpose_msc_brightfield')
CLASSIFIER_PATH = os.path.join(ROOT, 'checkpoints', 'classifier', 'classifier_v2.pth')

# Output: SEPARATE folder
OUTPUT_DIR = os.path.join(ROOT, 'data', 'pseudo_labeled_v2')
os.makedirs(os.path.join(OUTPUT_DIR, 'young'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'senescent'), exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_CELL_SIZE = 50
MAX_CELL_SIZE = 1500
CONFIDENCE_THRESHOLD = 0.95
CROP_SIZE = 512

# ==========================================
# 1. Load Cellpose model
# ==========================================
print("Loading Cellpose fine-tuned model...")
cp_model = models.CellposeModel(gpu=True, pretrained_model=CELLPOSE_MODEL)

# ==========================================
# 2. Load Classifier
# IMPORTANT: senescent=0, young=1 (ImageFolder alphabetical order)
# ==========================================
print("Loading classifier...")
from models.classifier.classifier import Classifier
classifier = Classifier(output_size=2)
ckpt = torch.load(CLASSIFIER_PATH, map_location='cpu')
if 'model_state_dict' in ckpt:
    classifier.load_state_dict(ckpt['model_state_dict'])
else:
    classifier.load_state_dict(ckpt)
classifier.to(DEVICE)
classifier.eval()

# Classifier was trained with ImageNet normalization
classify_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FIXED: correct label mapping
CLASS_NAMES = {0: 'senescent', 1: 'young'}

# ==========================================
# 3. Process each unlabeled TIF
# ==========================================
tif_files = sorted([f for f in os.listdir(UNLABELED_DIR) if f.lower().endswith('.tif')])
print(f"\nProcessing {len(tif_files)} unlabeled TIF images...")

total_cells = 0
total_kept = 0
class_counts = {'young': 0, 'senescent': 0}
class_areas = {'young': [], 'senescent': []}

for tif_file in tqdm(tif_files, desc="Segmenting"):
    img_path = os.path.join(UNLABELED_DIR, tif_file)
    img = np.array(Image.open(img_path).convert('RGB'))
    H, W = img.shape[:2]
    
    # Cellpose segmentation
    masks, flows, styles = cp_model.eval(
        img, diameter=None, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0
    )
    
    num_cells = masks.max()
    
    for cell_id in range(1, num_cells + 1):
        cell_mask = (masks == cell_id)
        
        # Get bounding box
        ys, xs = np.where(cell_mask)
        if len(ys) == 0:
            continue
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        bbox_h = y_max - y_min
        bbox_w = x_max - x_min
        
        # Filter by size
        if bbox_h < MIN_CELL_SIZE or bbox_w < MIN_CELL_SIZE:
            continue
        if bbox_h > MAX_CELL_SIZE or bbox_w > MAX_CELL_SIZE:
            continue
        
        total_cells += 1
        
        # ================================================
        # FIXED: 512x512 window centered on cell, NO ZOOM
        # ================================================
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        half = CROP_SIZE // 2  # 256
        
        # Calculate crop window
        crop_y1 = center_y - half
        crop_y2 = center_y + half
        crop_x1 = center_x - half
        crop_x2 = center_x + half
        
        # Shift if near edges
        if crop_y1 < 0:
            crop_y2 -= crop_y1  # shift down
            crop_y1 = 0
        if crop_y2 > H:
            crop_y1 -= (crop_y2 - H)  # shift up
            crop_y2 = H
        if crop_x1 < 0:
            crop_x2 -= crop_x1  # shift right
            crop_x1 = 0
        if crop_x2 > W:
            crop_x1 -= (crop_x2 - W)  # shift left
            crop_x2 = W
        
        # Clamp again (in case image is smaller than 512)
        crop_y1 = max(0, crop_y1)
        crop_x1 = max(0, crop_x1)
        
        # Crop directly from original image
        cell_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # If somehow smaller than 512x512 (edge case), pad with image mean
        if cell_crop.shape[0] < CROP_SIZE or cell_crop.shape[1] < CROP_SIZE:
            canvas = np.full((CROP_SIZE, CROP_SIZE, 3), 
                           cell_crop.mean(axis=(0,1)).astype(np.uint8), dtype=np.uint8)
            canvas[:cell_crop.shape[0], :cell_crop.shape[1]] = cell_crop
            cell_crop = canvas
        
        cell_pil = Image.fromarray(cell_crop)
        
        # Classify
        input_tensor = classify_transform(cell_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_class = probs.max(dim=1)
        
        confidence = confidence.item()
        pred_class = pred_class.item()
        class_name = CLASS_NAMES[pred_class]  # FIXED: correct mapping
        
        # Only keep high-confidence predictions
        if confidence < CONFIDENCE_THRESHOLD:
            continue
        
        # Calculate cell surface area directly from the Cellpose mask
        cell_area = np.sum(cell_mask)
        
        total_kept += 1
        class_counts[class_name] += 1
        class_areas[class_name].append(cell_area)
        
        # Save directly (already 512x512, no resize needed)
        save_name = f"pseudo_{tif_file.replace('.tif', '')}_{cell_id:03d}.png"
        # cell_pil.save(os.path.join(OUTPUT_DIR, class_name, save_name))  # <-- SAVING DISABLED

print(f"\n{'='*60}")
print(f"PSEUDO-LABELING RESULTS v2")
print(f"{'='*60}")
print(f"Total cells detected:    {total_cells}")
print(f"Kept (>{CONFIDENCE_THRESHOLD*100:.0f}% conf):    {total_kept}")
print(f"  Young:                 {class_counts['young']}")
print(f"  Senescent:             {class_counts['senescent']}")
print(f"Rejection rate:          {(1-total_kept/max(total_cells,1))*100:.1f}%\n")

avg_young_area = np.mean(class_areas['young']) if class_areas['young'] else 0
avg_senes_area = np.mean(class_areas['senescent']) if class_areas['senescent'] else 0
print(f"MORPHOLOGICAL ANALYSIS (Cellpose Mask Area):")
print(f"  Avg Young Area:      {avg_young_area:,.0f} pixels²")
print(f"  Avg Senescent Area:  {avg_senes_area:,.0f} pixels²")
if avg_young_area > 0:
    print(f"  Size Difference:     {avg_senes_area/avg_young_area:.1f}x larger")

print(f"\nSaved to: {OUTPUT_DIR}")
print(f"{'='*60}")
