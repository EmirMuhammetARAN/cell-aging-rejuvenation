import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import lpips
import warnings

# Skimage for SSIM
try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-image"])
    from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

# Paths
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')

CYCLEGAN_AGING = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2_sweep', 'epoch_160', 'aging')
CYCLEGAN_REJU = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2_sweep', 'epoch_160', 'rejuvenation')
LDM_AGING = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026', 'aging')
LDM_REJU = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026', 'rejuv')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load LPIPS
print("Loading LPIPS (AlexNet)...")
loss_fn_alex = lpips.LPIPS(net='alex').to(device)
loss_fn_alex.eval()

# Transform for LPIPS (requires [-1, 1] range)
lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def calculate_metrics(input_dir, generated_dir, desc=""):
    img_names = os.listdir(input_dir)
    
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    
    for img_name in tqdm(img_names, desc=desc):
        in_path = os.path.join(input_dir, img_name)
        gen_path = os.path.join(generated_dir, img_name)
        
        if not os.path.exists(gen_path):
            continue
            
        # Load for SSIM (numpy arrays, grayscale or RGB. We use RGB)
        img_in_pil = Image.open(in_path).convert('RGB')
        img_gen_pil = Image.open(gen_path).convert('RGB')
        
        img_in_np = np.array(img_in_pil)
        img_gen_np = np.array(img_gen_pil)
        
        # Calculate SSIM (channel_axis=-1 for RGB)
        s = ssim(img_in_np, img_gen_np, channel_axis=-1, data_range=255)
        total_ssim += s
        
        # Calculate LPIPS
        img_in_t = lpips_transform(img_in_pil).unsqueeze(0).to(device)
        img_gen_t = lpips_transform(img_gen_pil).unsqueeze(0).to(device)
        
        with torch.no_grad():
            l = loss_fn_alex(img_in_t, img_gen_t).item()
        total_lpips += l
        
        count += 1
        
    if count == 0:
        return 0, 0
    
    return total_ssim / count, total_lpips / count


def main():
    print("==========================================================")
    print("CALCULATING STRUCTURAL PRESERVATION METRICS (SSIM & LPIPS)")
    print("==========================================================")
    print("Note: SSIM (Higher is better, max 1.0), LPIPS (Lower is better, min 0.0)\n")
    
    c_age_ssim, c_age_lpips = calculate_metrics(TEST_YOUNG, CYCLEGAN_AGING, "CycleGAN Aging")
    c_rej_ssim, c_rej_lpips = calculate_metrics(TEST_SENES, CYCLEGAN_REJU, "CycleGAN Rejuv")
    
    l_age_ssim, l_age_lpips = calculate_metrics(TEST_YOUNG, LDM_AGING, "LDM Aging     ")
    l_rej_ssim, l_rej_lpips = calculate_metrics(TEST_SENES, LDM_REJU, "LDM Rejuv     ")
    
    print("\n\n================ FINAL RESULTS ================")
    print("TASK         | MODEL      | SSIM (UP) | LPIPS (DOWN)")
    print("-----------------------------------------------")
    print(f"Aging        | CycleGAN   | {c_age_ssim:.4f}  | {c_age_lpips:.4f}")
    print(f"Aging        | LDM        | {l_age_ssim:.4f}  | {l_age_lpips:.4f}")
    print("-----------------------------------------------")
    print(f"Rejuvenation | CycleGAN   | {c_rej_ssim:.4f}  | {c_rej_lpips:.4f}")
    print(f"Rejuvenation | LDM        | {l_rej_ssim:.4f}  | {l_rej_lpips:.4f}")
    print("===============================================")

if __name__ == "__main__":
    main()
