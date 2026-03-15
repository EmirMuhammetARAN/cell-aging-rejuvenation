import os
import sys

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, root_dir)

from src.data.tight_cropper import TightCropper
from src.data.parser import Parser

RAW_PATH = os.path.join(root_dir, "data", "raw")
PROCESSED_PATH = os.path.join(root_dir, "data", "processed_v5")
CROP_SIZE = 512
PADDING_RATIO = 0.3  # %30 padding, v4 ile aynı

SUBSETS = ["train", "val", "test"]

for subset in SUBSETS:
    subset_raw_path = os.path.join(RAW_PATH, subset)
    subset_processed_path = os.path.join(PROCESSED_PATH, subset)

    json_path = os.path.join(subset_raw_path, "via_project.json")

    if os.path.exists(json_path):
        parser = Parser(json_path)
        parsed_cells = parser.parse()
        print(f"\n[{subset}] {len(parsed_cells)} hücre parse edildi")

        cropper = TightCropper(subset_raw_path, subset_processed_path, CROP_SIZE, PADDING_RATIO)
        cropper.crop(parsed_cells)
    else:
        print(f"Warning: {json_path} not found. Skipping {subset}.")

print("\n✓ processed_v5 tamamlandı!")
