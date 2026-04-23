import os, sys
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.ldm.model_lpips import CellLDM

TEST_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v2','test','young')
TEST_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v2','test','senescent')

# AGA DİKKAT: Çıktıları ana klasör kirlenmesin diye ayrı bir test klasörüne alıyoruz
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'grid_search_ldm_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

model = CellLDM(num_classes=2)
model.to('cuda', memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)
model.unet.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v2_lpips_ft.pt'))['unet_state_dict'])
model.init_ema()
model.ema_unet.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v2_lpips_ft.pt'))['ema_unet_state_dict'])
model.eval()

# ---------------------------------------------------------
# GRID SEARCH PARAMETRELERİ (MÜHENDİSLİK KONTROL PANELİ)
# ---------------------------------------------------------
TEST_SAYISI = 10
STEPS_TRANS = 50   # Çeviri için adım sayısı
STEPS_RANDOM = 75  # Sıfırdan üretim için adım sayısı

# Denenecek Kombinasyonlar
strength_values = [0.60, 0.65, 0.70]
cfg_values_trans = [2.0, 3.0, 4.0]
cfg_values_random = [2.0, 3.0, 4.0, 5.0]

# Klasördeki ilk 10 resmi listeye alıyoruz
young_images = os.listdir(TEST_YOUNG_DIR)[:TEST_SAYISI]
senes_images = os.listdir(TEST_SENESCENT_DIR)[:TEST_SAYISI]


# =========================================================
# 1. ÇEVİRİ (TRANSLATION) TESTLERİ: AGING & REJUVENATION
# =========================================================
print("==================================================")
print("1. ÇEVİRİ (TRANSLATION) GRID SEARCH BAŞLIYOR...")
print("==================================================")

for strength in strength_values:
    for cfg in cfg_values_trans:
        print(f"\n[>>>] Çeviri Testi: Strength={strength}, CFG={cfg}")
        
        aging_dir = os.path.join(OUTPUT_DIR, f'aging_str_{strength}_cfg_{cfg}')
        rejuv_dir = os.path.join(OUTPUT_DIR, f'rejuv_str_{strength}_cfg_{cfg}')
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(rejuv_dir, exist_ok=True)

        # AGING (Genç -> Yaşlı)
        for img in young_images:
            out_path = os.path.join(aging_dir, img)
            if os.path.exists(out_path): continue  # Resim varsa atla
            
            img_path = os.path.join(TEST_YOUNG_DIR, img)
            image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
            with torch.no_grad():
                generated = model.translate(image, target_labels=torch.tensor([1], device='cuda'), strength=strength, num_steps=STEPS_TRANS, use_ema=True, guidance_scale=cfg)
            save_image(generated.cpu(), out_path)
        
        # REJUVENATION (Yaşlı -> Genç)
        for img in senes_images:
            out_path = os.path.join(rejuv_dir, img)
            if os.path.exists(out_path): continue  # Resim varsa atla
            
            img_path = os.path.join(TEST_SENESCENT_DIR, img)
            image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
            with torch.no_grad():
                generated = model.translate(image, target_labels=torch.tensor([0], device='cuda'), strength=strength, num_steps=STEPS_TRANS, use_ema=True, guidance_scale=cfg)
            save_image(generated.cpu(), out_path)


# =========================================================
# 2. SIFIRDAN ÜRETİM (RANDOM GENERATION) TESTLERİ
# =========================================================
print("\n==================================================")
print("2. RANDOM GENERATION GRID SEARCH BAŞLIYOR...")
print("==================================================")

for cfg in cfg_values_random:
    print(f"\n[>>>] Random Testi: CFG={cfg}")
    
    rand_young_dir = os.path.join(OUTPUT_DIR, f'random_young_cfg_{cfg}')
    rand_senes_dir = os.path.join(OUTPUT_DIR, f'random_senes_cfg_{cfg}')
    os.makedirs(rand_young_dir, exist_ok=True)
    os.makedirs(rand_senes_dir, exist_ok=True)
    
    # Genç Üretim (Label = 0)
    for i in range(TEST_SAYISI):
        out_path = os.path.join(rand_young_dir, f'young_sample_{i}.png')
        if os.path.exists(out_path): continue  # Resim varsa atla
        
        label_tensor = torch.zeros((1,), dtype=torch.long, device='cuda')
        with torch.no_grad():
            generated = model.sample(num_samples=1, device='cuda', labels=label_tensor, use_ema=True, guidance_scale=cfg, num_steps=STEPS_RANDOM)
        save_image(generated.cpu(), out_path)

    # Yaşlı Üretim (Label = 1)
    for i in range(TEST_SAYISI):
        out_path = os.path.join(rand_senes_dir, f'senescent_sample_{i}.png')
        if os.path.exists(out_path): continue  # Resim varsa atla
        
        label_tensor = torch.ones((1,), dtype=torch.long, device='cuda')
        with torch.no_grad():
            generated = model.sample(num_samples=1, device='cuda', labels=label_tensor, use_ema=True, guidance_scale=cfg, num_steps=STEPS_RANDOM)
        save_image(generated.cpu(), out_path)

print("\n[✓] Bütün Grid Search testleri tamamlandı aga! 'grid_search_ldm_v2' klasörünü inceleyebilirsin.")