import os
import sys

# Get root directory
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, root_dir)

from src.data.cropper import Cropper
from src.data.parser import Parser

RAW_PATH = os.path.join(root_dir, "data", "unlabeled")
PROCESSED_PATH = os.path.join(root_dir, "data", "pseudo_label")
CROP_SIZE = 512

json_path = os.path.join(RAW_PATH, "via_project.json")

if os.path.exists(json_path):
    parser = Parser(json_path)
    parsed_cells = parser.parse()
    print(f"Parsed {len(parsed_cells)} pseudo-labeled cells from {RAW_PATH}")

    cropper = Cropper(RAW_PATH, PROCESSED_PATH, CROP_SIZE)
    cropper.crop(parsed_cells)
    print("\n[OK] Pseudo-label data prepared successfully!")
else:
    print(f"Error: {json_path} not found. Please run auto_annotate.py first.")
