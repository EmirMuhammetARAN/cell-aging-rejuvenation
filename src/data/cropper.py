import os 
from PIL import Image
import numpy as np


class Cropper:
    def __init__(self, raw_path, processed_path, crop_size=512):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.crop_size = crop_size
        os.makedirs(os.path.join(self.processed_path, "young"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "senescent"), exist_ok=True)

    def crop(self, parsed_cells):
        counter = 0
        for cell in parsed_cells:
            filename = cell['filename']
            points_x = list(cell['points_x'])
            points_y = list(cell['points_y'])
            label = cell['label']

            img = Image.open(os.path.join(self.raw_path, filename))
            mid_x = (max(points_x) + min(points_x)) // 2
            mid_y = (max(points_y) + min(points_y)) // 2

            left = mid_x - self.crop_size // 2
            top  = mid_y - self.crop_size // 2

            left = max(0, min(left, img.width  - self.crop_size))
            top  = max(0, min(top,  img.height - self.crop_size))

            right  = left + self.crop_size
            bottom = top  + self.crop_size

            region = img.crop((left, top, right, bottom))
            canvas = region

            canvas.save(os.path.join(self.processed_path, label, os.path.splitext(filename)[0] + f"_{counter}.jpg"))
            counter += 1