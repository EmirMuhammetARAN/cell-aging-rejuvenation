"""
Calculate Input-Output Structural Preservation Metrics (Table 4.3):
SSIM (Higher is better) & LPIPS (Lower is better)

Evaluates:
  1. CycleGAN (Epoch 160, Seed 2026)
  2. CUT GAN (Epoch 200, Seed 2026)
  3. LDM Low-FID (Seed 2026: Aging s=0.75, cfg=4.0 | Rejuv s=0.65, cfg=3.5)
  4. LDM High-ACC (100 steps: Aging s=0.80, cfg=6.0 | Rejuv s=0.70, cfg=5.0)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import lpips
import warnings
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test source ground truth cohorts
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')

# Model output paths
MODELS = {
    'CycleGAN (Ep 160, Seed 2026)': {
        'aging': os.path.join(root_dir, 'results', 'generated', 'cyclegan_seed_sweep', 'seed_2026', 'aging'),
        'rejuv': os.path.join(root_dir, 'results', 'generated', 'cyclegan_seed_sweep', 'seed_2026', 'rejuv'),
    },
    'CUT GAN (Ep 200, Seed 2026)': {
        'aging': os.path.join(root_dir, 'results', 'generated', 'cut_seed_sweep', 'seed_2026', 'aging'),
        'rejuv': os.path.join(root_dir, 'results', 'generated', 'cut_seed_sweep', 'seed_2026', 'rejuv'),
    },
    'LDM Low-FID (Seed 2026)': {
        'aging': os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026', 'aging'),
        'rejuv': os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026', 'rejuv'),
    },
    'LDM High-ACC (100 steps)': {
        'aging': os.path.join(root_dir, 'results', 'ablation_masked_vs_standard', 'strictly_paired_100steps', 'high_acc', 'aging_standard'),
        'rejuv': os.path.join(root_dir, 'results', 'ablation_masked_vs_standard', 'strictly_paired_100steps', 'high_acc', 'rejuvenation_standard'),
    }
}

# Load LPIPS (AlexNet)
print("[*] Loading LPIPS (AlexNet)...")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
loss_fn_alex.eval()

lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def calculate_metrics(input_dir, generated_dir, desc=""):
    if not os.path.exists(generated_dir):
        print(f"Directory not found: {generated_dir}")
        return 0.0, 0.0, 0

    img_names = sorted(os.listdir(input_dir))
    
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    for img_name in tqdm(img_names, desc=desc, leave=False):
        in_path = os.path.join(input_dir, img_name)
        gen_path = os.path.join(generated_dir, img_name)
        
        if not os.path.exists(gen_path):
            continue
            
        img_in_pil = Image.open(in_path).convert('RGB')
        img_gen_pil = Image.open(gen_path).convert('RGB')
        
        # Ensure sizes match
        if img_in_pil.size != img_gen_pil.size:
            img_gen_pil = img_gen_pil.resize(img_in_pil.size, Image.Resampling.LANCZOS)
        
        img_in_np = np.array(img_in_pil)
        img_gen_np = np.array(img_gen_pil)
        
        # SSIM
        s = ssim(img_in_np, img_gen_np, channel_axis=-1, data_range=255)
        total_ssim += s
        
        # LPIPS
        img_in_t = lpips_transform(img_in_pil).unsqueeze(0).to(device)
        img_gen_t = lpips_transform(img_gen_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            l = loss_fn_alex(img_in_t, img_gen_t).item()
        total_lpips += l
        
        count += 1
        
    if count == 0:
        return 0.0, 0.0, 0
    
    return total_ssim / count, total_lpips / count, count


def main():
    print("=" * 80)
    print(" CALCULATING TABLE 4.3: INPUT-OUTPUT STRUCTURAL PRESERVATION METRICS ")
    print("=" * 80)

    results = []

    for model_name, paths in MODELS.items():
        print(f"\nEvaluating: {model_name}")
        
        # Aging: Young -> Senescent
        ssim_a, lpips_a, cnt_a = calculate_metrics(TEST_YOUNG, paths['aging'], f"Aging ({model_name})")
        print(f"  Aging:        SSIM = {ssim_a:.4f} | LPIPS = {lpips_a:.4f} (N={cnt_a})")
        
        # Rejuvenation: Senescent -> Young
        ssim_r, lpips_r, cnt_r = calculate_metrics(TEST_SENES, paths['rejuv'], f"Rejuv ({model_name})")
        print(f"  Rejuvenation: SSIM = {ssim_r:.4f} | LPIPS = {lpips_r:.4f} (N={cnt_r})")
        
        results.append({
            'model': model_name,
            'aging_ssim': ssim_a,
            'aging_lpips': lpips_a,
            'aging_n': cnt_a,
            'rejuv_ssim': ssim_r,
            'rejuv_lpips': lpips_r,
            'rejuv_n': cnt_r,
        })

    print("\n" + "=" * 90)
    print("TABLE 4.3 FORMATTED RESULTS:")
    print("=" * 90)
    print(f"{'Direction':<22} | {'Model':<30} | {'SSIM ↑':<8} | {'LPIPS ↓':<8} | {'Interpretation'}")
    print("-" * 90)

    for r in results:
        interp_a = "Greater similarity to input" if r['aging_ssim'] > 0.85 else "Lower similarity (phenotypic restructuring)"
        print(f"{'Young → Senescent':<22} | {r['model']:<30} | {r['aging_ssim']:<8.4f} | {r['aging_lpips']:<8.4f} | {interp_a}")

    print("-" * 90)
    for r in results:
        interp_r = "Greater similarity to input" if r['rejuv_ssim'] > 0.85 else "Lower similarity (phenotypic restructuring)"
        print(f"{'Senescent → Young':<22} | {r['model']:<30} | {r['rejuv_ssim']:<8.4f} | {r['rejuv_lpips']:<8.4f} | {interp_r}")
    print("=" * 90)


if __name__ == '__main__':
    main()
