import os
import sys
import torch
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
# Metric fonksiyonlarını önceki scriptten import ediyoruz
from src.inference.ldm.calculate_metrics import calculate_fid, calculate_classifier_score, TEST_YOUNG_DIR, TEST_SENESCENT_DIR

SEEDS = [42, 1024, 2026, 7777, 9999]
# En istikrarlı performans gösteren epoch 50'yi kullanıyoruz, istersen _last.pt olarak değiştirebilirsin
CKPT_PATH = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'seed_sweep')

AGING_STRENGTH = 0.75
AGING_CFG = 4.0

REJUV_STRENGTH = 0.65
REJUV_CFG = 3.5

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def run_seed_sweep():
    # Model Setup
    model = CellLDM(num_classes=2)
    model.to('cuda', memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    
    print(f"Yükleniyor: {os.path.basename(CKPT_PATH)}")
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    model.unet.load_state_dict(checkpoint['unet_state_dict'])
    model.init_ema()
    if 'ema_unet_state_dict' in checkpoint:
        model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    young_images = os.listdir(TEST_YOUNG_DIR)
    senes_images = os.listdir(TEST_SENESCENT_DIR)

    results = []

    for seed in SEEDS:
        print(f"\n==============================================")
        print(f"[{seed}] TESTING SEED {seed}")
        print(f"==============================================")
        set_seed(seed)
        
        seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")
        aging_dir = os.path.join(seed_dir, 'aging')
        rejuv_dir = os.path.join(seed_dir, 'rejuv')
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(rejuv_dir, exist_ok=True)
        
        # AGING
        print(f"[{seed}] Generating Aging...")
        for img_name in tqdm(young_images, desc=f"Aging (Seed {seed})"):
            out_path = os.path.join(aging_dir, img_name)
            if os.path.exists(out_path): continue
            
            img_path = os.path.join(TEST_YOUNG_DIR, img_name)
            image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
            with torch.no_grad():
                generated = model.translate(image, target_labels=torch.tensor([1], device='cuda'), strength=AGING_STRENGTH, num_steps=100, use_ema=True, guidance_scale=AGING_CFG)
            save_image(generated.cpu(), out_path)
            
        # REJUVENATION
        print(f"[{seed}] Generating Rejuvenation...")
        for img_name in tqdm(senes_images, desc=f"Rejuv (Seed {seed})"):
            out_path = os.path.join(rejuv_dir, img_name)
            if os.path.exists(out_path): continue
            
            img_path = os.path.join(TEST_SENESCENT_DIR, img_name)
            image = transform(Image.open(img_path)).unsqueeze(0).to('cuda', memory_format=torch.channels_last)
            with torch.no_grad():
                generated = model.translate(image, target_labels=torch.tensor([0], device='cuda'), strength=REJUV_STRENGTH, num_steps=100, use_ema=True, guidance_scale=REJUV_CFG)
            save_image(generated.cpu(), out_path)
            
        torch.cuda.empty_cache()
        
        print(f"[{seed}] Calculating Metrics...")
        aging_fid = calculate_fid(TEST_SENESCENT_DIR, aging_dir)
        rejuv_fid = calculate_fid(TEST_YOUNG_DIR, rejuv_dir)
        aging_fool_acc = calculate_classifier_score(aging_dir, target_label=0) # senescent=0
        rejuv_fool_acc = calculate_classifier_score(rejuv_dir, target_label=1) # young=1
        
        print(f"Seed: {seed} | AGING FID: {aging_fid:.2f} | AGING %: {aging_fool_acc:.2f} | REJUV FID: {rejuv_fid:.2f} | REJUV %: {rejuv_fool_acc:.2f}")
        
        results.append({
            'seed': seed,
            'aging_fid': aging_fid,
            'rejuv_fid': rejuv_fid,
            'aging_fool_acc': aging_fool_acc,
            'rejuv_fool_acc': rejuv_fool_acc
        })
        
    print("\n================ FINAL RESULTS ================")
    print(f"{'SEED':<6} | {'AGING FID':<9} | {'AGING %':<7} | {'REJUV FID':<9} | {'REJUV %':<7}")
    print("-" * 55)
    for r in results:
        print(f"{r['seed']:<6} | {r['aging_fid']:<9.2f} | {r['aging_fool_acc']:<7.2f} | {r['rejuv_fid']:<9.2f} | {r['rejuv_fool_acc']:<7.2f}")
        
    print("\n================ SENSITIVITY (VARIANCE) ANALYSIS ================")
    a_fids = [r['aging_fid'] for r in results]
    a_accs = [r['aging_fool_acc'] for r in results]
    r_fids = [r['rejuv_fid'] for r in results]
    r_accs = [r['rejuv_fool_acc'] for r in results]
    
    print(f"AGING FID: Mean = {np.mean(a_fids):.2f}  |  Std Dev (Sensitivity) = ±{np.std(a_fids):.2f}")
    print(f"AGING ACC: Mean = {np.mean(a_accs):.2f}% |  Std Dev (Sensitivity) = ±{np.std(a_accs):.2f}%")
    print(f"REJUV FID: Mean = {np.mean(r_fids):.2f}  |  Std Dev (Sensitivity) = ±{np.std(r_fids):.2f}")
    print(f"REJUV ACC: Mean = {np.mean(r_accs):.2f}% |  Std Dev (Sensitivity) = ±{np.std(r_accs):.2f}%")
    
    best_aging_fid = min(results, key=lambda x: x['aging_fid'])
    best_rejuv_fid = min(results, key=lambda x: x['rejuv_fid'])
    
    print("\n🏆 THE LUCKY SEEDS 🏆")
    print(f"Lowest AGING FID Seed: {best_aging_fid['seed']} (FID: {best_aging_fid['aging_fid']:.2f})")
    print(f"Lowest REJUV FID Seed: {best_rejuv_fid['seed']} (FID: {best_rejuv_fid['rejuv_fid']:.2f})")

if __name__ == "__main__":
    run_seed_sweep()
