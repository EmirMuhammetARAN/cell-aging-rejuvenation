import os
import sys

# Get root directory
current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, root_dir)

from src.data.cropper import Cropper
from src.data.parser import Parser

RAW_PATH = os.path.join(root_dir, "data", "raw")
PROCESSED_PATH = os.path.join(root_dir, "data", "processed_v2")
CROP_SIZE = 512

SUBSETS = ["train", "val", "test"]

for subset in SUBSETS:
    subset_raw_path = os.path.join(RAW_PATH, subset)
    subset_processed_path = os.path.join(PROCESSED_PATH, subset)

    json_path = os.path.join(subset_raw_path, "via_project.json")

    if os.path.exists(json_path):
        parser = Parser(json_path)
        parsed_cells = parser.parse()
        print(f"Parsed {len(parsed_cells)} cells from {json_path}")

        cropper = Cropper(subset_raw_path, subset_processed_path, CROP_SIZE)
        cropper.crop(parsed_cells)
        print(f"Cropped and saved cells for {subset} subset.")
    else:
        print(f"Warning: {json_path} not found. Skipping {subset} subset.")