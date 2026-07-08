import os, sys
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import glob

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.ldm.model_lpips import CellLDM

# Parametreler (Kullanıcının Belirlediği En İyi Ayarlar)
AGING_STRENGTH = 0.75
AGING_CFG = 4.0

REJUV_STRENGTH = 0.65
REJUV_CFG = 3.5

EXPERIMENT_NAME = "v12_v4_data_lpips"

TEST_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'checkpoint_comparison_lpips')

os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Test resimlerini seç (TÜMÜ)
young_images = os.listdir(TEST_YOUNG_DIR)
senes_images = os.listdir(TEST_SENESCENT_DIR)

# Checkpointleri bul
ckpt_dir = os.path.join(root_dir, 'checkpoints', 'ldm')
# best_model'i, v4 best'i (kıyaslama için) ve LPIPS epoch'larını listeye ekle
checkpoints_to_test = [
    os.path.join(ckpt_dir, f'best_model_{EXPERIMENT_NAME}.pt'),
    os.path.join(ckpt_dir, 'best_model_v12_v4_data.pt')
]
for epoch in [25, 50, 'last']:
    if epoch == 'last':
        path = os.path.join(ckpt_dir, f'checkpoint_{EXPERIMENT_NAME}_{epoch}.pt')
    else:
        path = os.path.join(ckpt_dir, f'checkpoint_{EXPERIMENT_NAME}_epoch_{epoch}.pt')
        
    if os.path.exists(path):
        checkpoints_to_test.append(path)

model = CellLDM(num_classes=2)
model.to('cuda', memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)

print("==================================================")
print(f"FULL CHECKPOINT KIYASLAMA TESTİ BAŞLIYOR (Toplam {len(checkpoints_to_test)} model)")
print(f"Toplam Veri: {len(young_images)} Young, {len(senes_images)} Senescent")
print(f"Aging Ayarları: Strength={AGING_STRENGTH}, CFG={AGING_CFG}")
print(f"Rejuv Ayarları: Strength={REJUV_STRENGTH}, CFG={REJUV_CFG}")
print("==================================================\n")

for ckpt_path in checkpoints_to_test:
    ckpt_name = os.path.basename(ckpt_path).replace('.pt', '')
    print(f"\n[>>>] Yükleniyor: {ckpt_name}")
    
    # Modeli yükle
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.unet.load_state_dict(checkpoint['unet_state_dict'])
    model.init_ema()
    if 'ema_unet_state_dict' in checkpoint:
        model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])
    model.eval()
    
    # Çıktı klasörleri
    ckpt_out_dir = os.path.join(OUTPUT_DIR, ckpt_name)
    aging_dir = os.path.join(ckpt_out_dir, 'aging')
    rejuv_dir = os.path.join(ckpt_out_dir, 'rejuv')
    os.makedirs(aging_dir, exist_ok=True)
    os.makedirs(rejuv_dir, exist_ok=True)
    
    # AGING
    print(f"      -> Yaşlandırma üretiliyor ({len(young_images)} adet)...")
    for img_name in tqdm(young_images, desc=f"Aging ({ckpt_name})"):
        out_path = os.path.join(aging_dir, img_name)
        if os.path.exists(out_path): continue # Eğer kesilip devam ederse atlar
        
        img_path = os.path.join(TEST_YOUNG_DIR, img_name)
        image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
        with torch.no_grad():
            generated = model.translate(image, target_labels=torch.tensor([1], device='cuda'), strength=AGING_STRENGTH, num_steps=50, use_ema=True, guidance_scale=AGING_CFG)
        
        # FID / LPIPS hesaplanacağı için SADECE üretilen resmi kaydediyoruz
        save_image(generated.cpu(), out_path)
        
    # REJUVENATION
    print(f"      -> Gençleştirme üretiliyor ({len(senes_images)} adet)...")
    for img_name in tqdm(senes_images, desc=f"Rejuv ({ckpt_name})"):
        out_path = os.path.join(rejuv_dir, img_name)
        if os.path.exists(out_path): continue # Eğer kesilip devam ederse atlar
        
        img_path = os.path.join(TEST_SENESCENT_DIR, img_name)
        image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
        with torch.no_grad():
            generated = model.translate(image, target_labels=torch.tensor([0], device='cuda'), strength=REJUV_STRENGTH, num_steps=50, use_ema=True, guidance_scale=REJUV_CFG)
        
        # FID / LPIPS hesaplanacağı için SADECE üretilen resmi kaydediyoruz
        save_image(generated.cpu(), out_path)
        
    torch.cuda.empty_cache()

print("\nTÜM TESTLER TAMAMLANDI! Çıktılar: results/generated/checkpoint_comparison_full/")
