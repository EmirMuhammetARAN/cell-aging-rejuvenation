"""
Identity Preservation Evaluation: SSIM + LPIPS
Compares INPUT images vs GENERATED outputs to measure structural preservation.
"""
import os, sys, torch, gc
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import lpips

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, root_dir)

# ============================================
# CONFIG
# ============================================
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')

# Generated images directories
CYCLEGAN_AGING = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2_sweep', 'epoch_110', 'aging')
CYCLEGAN_REJU  = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2_sweep', 'epoch_110', 'rejuvenation')

# Find LDM sweep directory
LDM_SWEEP_BASE = os.path.join(root_dir, 'results', 'generated', 'ldm')
ldm_dirs = [d for d in os.listdir(LDM_SWEEP_BASE) if d.startswith('sweep_')]
# Find the optimal one: as0.8_ac5.0_rs0.7_rc4.0
optimal_tag = 'sweep_as0.8_ac5.0_rs0.7_rc4.0'
if optimal_tag not in ldm_dirs:
    # Try to find closest match
    for d in ldm_dirs:
        if 'as0.8' in d and 'ac5.0' in d:
            optimal_tag = d
            break
    else:
        optimal_tag = ldm_dirs[0] if ldm_dirs else None
        print(f"WARNING: Could not find optimal LDM dir, using: {optimal_tag}")

if optimal_tag:
    LDM_AGING = os.path.join(LDM_SWEEP_BASE, optimal_tag, 'aging')
    LDM_REJU  = os.path.join(LDM_SWEEP_BASE, optimal_tag, 'rejuvenation')
else:
    print("ERROR: No LDM sweep directories found!")
    sys.exit(1)

print(f"CycleGAN aging: {CYCLEGAN_AGING}")
print(f"CycleGAN reju:  {CYCLEGAN_REJU}")
print(f"LDM aging:      {LDM_AGING}")
print(f"LDM reju:       {LDM_REJU}")

# ============================================
# LPIPS model
# ============================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
lpips_fn = lpips.LPIPS(net='vgg').to(DEVICE)
lpips_fn.eval()

lpips_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def compute_metrics(input_dir, generated_dir, task_name):
    """Compute SSIM and LPIPS between input and generated images."""
    ssim_scores = []
    lpips_scores = []
    
    gen_files = sorted(os.listdir(generated_dir))
    
    for img_name in tqdm(gen_files, desc=task_name):
        input_path = os.path.join(input_dir, img_name)
        gen_path = os.path.join(generated_dir, img_name)
        
        if not os.path.exists(input_path):
            continue
        
        # Load images
        img_input = Image.open(input_path).convert('RGB')
        img_gen = Image.open(gen_path).convert('RGB')
        
        # Resize to same size if needed
        if img_input.size != img_gen.size:
            img_gen = img_gen.resize(img_input.size, Image.LANCZOS)
        
        # SSIM (on grayscale)
        inp_gray = np.array(img_input.convert('L'))
        gen_gray = np.array(img_gen.convert('L'))
        s = ssim(inp_gray, gen_gray, data_range=255)
        ssim_scores.append(s)
        
        # LPIPS
        inp_t = lpips_transform(img_input).unsqueeze(0).to(DEVICE)
        gen_t = lpips_transform(img_gen).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            d = lpips_fn(inp_t, gen_t).item()
        lpips_scores.append(d)
    
    return {
        'ssim_mean': np.mean(ssim_scores),
        'ssim_std': np.std(ssim_scores),
        'lpips_mean': np.mean(lpips_scores),
        'lpips_std': np.std(lpips_scores),
        'n': len(ssim_scores)
    }

# ============================================
# RUN
# ============================================
print(f"\n{'='*70}")
print("IDENTITY PRESERVATION EVALUATION")
print(f"{'='*70}\n")

results = {}

# CycleGAN Aging: young input -> senescent output
print("1. CycleGAN - Aging (young → senescent)")
results['cyc_aging'] = compute_metrics(TEST_YOUNG, CYCLEGAN_AGING, "CycleGAN Aging")

# CycleGAN Rejuvenation: senescent input -> young output
print("\n2. CycleGAN - Rejuvenation (senescent → young)")
results['cyc_reju'] = compute_metrics(TEST_SENES, CYCLEGAN_REJU, "CycleGAN Reju")

# LDM Aging
print("\n3. LDM - Aging (young → senescent)")
results['ldm_aging'] = compute_metrics(TEST_YOUNG, LDM_AGING, "LDM Aging")

# LDM Rejuvenation
print("\n4. LDM - Rejuvenation (senescent → young)")
results['ldm_reju'] = compute_metrics(TEST_SENES, LDM_REJU, "LDM Reju")

# Cleanup
del lpips_fn; torch.cuda.empty_cache(); gc.collect()

# ============================================
# RESULTS TABLE
# ============================================
print(f"\n{'='*70}")
print("RESULTS: Identity Preservation Metrics")
print(f"{'='*70}")
print(f"{'Task':<30} | {'SSIM ↑':>12} | {'LPIPS ↓':>12} | {'N':>5}")
print(f"{'-'*70}")

labels = {
    'cyc_aging': 'CycleGAN  Aging',
    'cyc_reju':  'CycleGAN  Rejuvenation',
    'ldm_aging': 'LDM       Aging',
    'ldm_reju':  'LDM       Rejuvenation',
}

for key, label in labels.items():
    r = results[key]
    print(f"{label:<30} | {r['ssim_mean']:.4f}±{r['ssim_std']:.4f} | {r['lpips_mean']:.4f}±{r['lpips_std']:.4f} | {r['n']:>5}")

print(f"\n{'='*70}")
print("INTERPRETATION:")
print("  SSIM  → Higher = more structure preserved (ideal: 0.6-0.85)")
print("  LPIPS → Lower  = more perceptually similar (ideal: 0.1-0.4)")
print("  Too high SSIM (>0.95) = identity mapping (no transformation)")
print("  Too low SSIM  (<0.4)  = structure destroyed")
print(f"{'='*70}")
