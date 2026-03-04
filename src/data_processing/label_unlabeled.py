import os
import sys
import numpy as np
from PIL import Image
import tifffile

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname((os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

UNLABELED_DATA_PATH = "data/unlabeled_data"
VAR_THRESHOLD = 50

for filename in os.listdir(UNLABELED_DATA_PATH):
    image = tifffile.imread(os.path.join(UNLABELED_DATA_PATH, filename))
    h, w, _ = image.shape
    y_positions = [0, 512, 1024, 1432]       
    x_positions = [0, 512, 1024, 1536, 2048]  

    for y in y_positions:
        for x in x_positions:
            patch = image[y:y+512, x:x+512]
            
            variance = np.var(patch)
            base = os.path.splitext(filename)[0]

            if variance < VAR_THRESHOLD:
                patch_filename = f"{base}_{y}_{x}_var{variance:.0f}.tif"
                patch_path = os.path.join("data/unlabeled_patches/rejected", patch_filename)
                os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                Image.fromarray(patch).save(patch_path)
            else:
                patchImage = Image.fromarray(patch)
                patch_filename = f"{base}_{y}_{x}.jpg"
                patch_path = os.path.join(f"data/unlabeled_patches/", patch_filename)
                os.makedirs(os.path.dirname(patch_path), exist_ok=True)
                Image.fromarray(patch).save(patch_path)