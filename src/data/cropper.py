import os 
from PIL import Image
import numpy as np

class Cropper:
    # target_cell_ratio yerine edge_padding (kenar boşluğu) ekledik
    def __init__(self, raw_path, processed_path, crop_size=512, edge_padding=10):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.crop_size = crop_size
        self.edge_padding = edge_padding # Kenarlardan bırakılacak piksel boşluğu
        os.makedirs(os.path.join(self.processed_path, "young"), exist_ok=True)
        os.makedirs(os.path.join(self.processed_path, "senescent"), exist_ok=True)

    def _get_dynamic_crop_size(self, points_x, points_y):
        cell_w = max(points_x) - min(points_x)
        cell_h = max(points_y) - min(points_y)
        cell_size = max(cell_w, cell_h) # Hücrenin en geniş yerini bul

        # Çerçeve boyutu: Hücrenin boyutu + (sağdan ve soldan 10'ar piksel boşluk)
        dynamic_crop = cell_size + (self.edge_padding * 2)

        # Eski koddaki min_crop sınırını kaldırdık ki küçük hücreler tam büyüsün.
        return dynamic_crop, cell_size

    def crop(self, parsed_cells):
        counter = 0
        skipped = 0

        for cell in parsed_cells:
            filename  = cell['filename']
            points_x  = list(cell['points_x'])
            points_y  = list(cell['points_y'])
            label     = cell['label']

            img = Image.open(os.path.join(self.raw_path, filename))
            img_w, img_h = img.size

            mid_x = (max(points_x) + min(points_x)) // 2
            mid_y = (max(points_y) + min(points_y)) // 2

            dynamic_crop, cell_size = self._get_dynamic_crop_size(points_x, points_y)

            # Görüntü kırpma alanından küçükse atla
            if img_w < dynamic_crop or img_h < dynamic_crop:
                print(f"  ⚠ Skip: {filename} görüntü çok küçük "
                      f"({img_w}x{img_h} < {dynamic_crop}px)")
                skipped += 1
                continue

            half = dynamic_crop // 2

            left   = mid_x - half
            top    = mid_y - half
            right  = mid_x + half
            bottom = mid_y + half

            # Taşıyorsa kaydır → hücre merkezde olmaz ama siyah boşluk da olmaz
            if left < 0:
                right -= left   
                left = 0
            if top < 0:
                bottom -= top
                top = 0
            if right > img_w:
                left -= (right - img_w)
                right = img_w
            if bottom > img_h:
                top -= (bottom - img_h)
                bottom = img_h

            # Burası önemli: Önce hücreye sıfıra sıfır (ufak bir payla) crop atıyor
            region = img.crop((left, top, right, bottom))
            # Sonra o kestiği boyutu (örneğin 50x50'yi) alıp 512x512'ye sündürüp büyütüyor
            region = region.resize((self.crop_size, self.crop_size), Image.LANCZOS)

            save_path = os.path.join(
                self.processed_path, label,
                os.path.splitext(filename)[0] + f"_{counter}.jpg"
            )
            region.save(save_path, quality=95)

            if counter % 100 == 0:
                print(f"  [{counter}] {filename} | "
                      f"hücre: {cell_size}px | "
                      f"crop: {dynamic_crop}px → {self.crop_size}px")
            counter += 1

        print(f"\nToplam: {counter} kaydedildi, {skipped} skip edildi")