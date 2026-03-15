import os
from PIL import Image
import numpy as np


class TightCropper:
    """
    Tight crop: sadece hedef hücrenin bbox'ı + padding.
    Maskeleme yok, komşu hücreler blur yok, arka plan doğal kalır.
    Output: processed_v5/
    """

    def __init__(self, raw_path, processed_path, crop_size=512, padding_ratio=0.3):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.crop_size = crop_size
        self.padding_ratio = padding_ratio
        os.makedirs(os.path.join(self.processed_path, "young"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "senescent"), exist_ok=True)

    def crop(self, parsed_cells):
        # Group cells by filename
        cells_by_file = {}
        for cell in parsed_cells:
            fname = cell['filename']
            if fname not in cells_by_file:
                cells_by_file[fname] = []
            cells_by_file[fname].append(cell)

        counter = 0
        for filename, cells in cells_by_file.items():
            img_path = os.path.join(self.raw_path, filename)
            if not os.path.exists(img_path):
                print(f"  ⚠ {filename} bulunamadı, atlanıyor")
                continue

            img = np.array(Image.open(img_path).convert('RGB'))
            h, w = img.shape[:2]

            for cell in cells:
                points_x = np.array(cell['points_x'])
                points_y = np.array(cell['points_y'])
                label = cell['label']

                # 1) Bounding box
                min_x, max_x = int(points_x.min()), int(points_x.max())
                min_y, max_y = int(points_y.min()), int(points_y.max())

                cell_w = max_x - min_x
                cell_h = max_y - min_y

                # 2) Make square (longer side)
                side = max(cell_w, cell_h)

                # 3) Add padding
                padding = int(side * self.padding_ratio)
                side = side + 2 * padding

                # 4) Minimum size
                side = max(side, 128)

                # 5) Center of bbox
                cx = (min_x + max_x) // 2
                cy = (min_y + max_y) // 2

                # 6) Crop coordinates
                left = cx - side // 2
                top = cy - side // 2
                right = left + side
                bottom = top + side

                # 7) Clamp to image bounds
                left = max(0, left)
                top = max(0, top)
                right = min(w, right)
                bottom = min(h, bottom)

                # 8) Crop directly from original image (no masking)
                cropped = img[top:bottom, left:right]

                # 9) Pad to square if edge-clipped
                ch, cw = cropped.shape[:2]
                if ch != cw:
                    target = max(ch, cw)
                    # Use image median as pad color to blend naturally
                    bg_color = np.median(img.reshape(-1, 3), axis=0).astype(np.uint8)
                    padded = np.full((target, target, 3), bg_color, dtype=np.uint8)
                    y_off = (target - ch) // 2
                    x_off = (target - cw) // 2
                    padded[y_off:y_off+ch, x_off:x_off+cw] = cropped
                    cropped = padded

                # 10) Resize to crop_size x crop_size
                result = Image.fromarray(cropped).resize(
                    (self.crop_size, self.crop_size), Image.LANCZOS
                )

                # Save
                out_path = os.path.join(
                    self.processed_path, label,
                    os.path.splitext(filename)[0] + f"_{counter}.jpg"
                )
                result.save(out_path, quality=95)
                counter += 1

        print(f"  ✓ {counter} hücre kaydedildi → {self.processed_path}")
