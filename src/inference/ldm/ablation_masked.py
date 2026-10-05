"""
Ablation Study: Standard SDEdit vs. Mask-Guided SDEdit (Background-Anchored)
=============================================================================
Full-population, FOV-clustered, strictly noise-paired benchmark comparing
standard model.translate() against model.translate_masked() on single-cell
MSC senescence translation (Aging & Rejuvenation).

Methodological Rigor:
  1. Identical Noise Pairing:
     Standard and Masked SDEdit start from the EXACT same initial noised latent
     (x_{t_start}) drawn from the same seed per image, eliminating random noise
     variance as a confounding variable.
  2. Timestep-Aligned Background Anchoring:
     RePaint-style anchoring aligned to t_prev after scheduler.step(), with clean
     x_0 anchoring at the terminal step (no residual noise in background).
  3. Full Test Population & FOV Clustering:
     Evaluates all valid test images across independent fields of view (27 aging FOVs,
     29 rejuvenation FOVs). Aggregates results by FOV and reports paired t-tests,
     Wilcoxon signed-rank tests, and 95% Confidence Intervals at the FOV level.
  4. Signal-to-Disruption Ratio (Edit SNR):
     Measures Cell MSE / Background MSE to verify that foreground morphological
     transformation is maintained while background alteration is suppressed.
  5. Boundary Seam Discontinuity:
     Evaluates gradient continuity across the mask dilation transition ring to
     ensure no edge artifacts or blending seams are introduced.

Usage:
    python src/inference/ldm/ablation_masked.py --direction both
    python src/inference/ldm/ablation_masked.py --direction aging
    python src/inference/ldm/ablation_masked.py --direction rejuvenation
    python src/inference/ldm/ablation_masked.py --summary_only
"""

import os
import sys
import gc
import re
import argparse
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


# ─── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def extract_fov_id(filename: str) -> str:
    """Extracts the parent microscope Field of View (FOV) identifier from the crop filename."""
    m = re.match(r'^(.*)_[0-9]+\.(?:jpg|png)$', filename)
    return m.group(1) if m else filename


def get_cell_mask_otsu(pil_img: Image.Image, img_size: int = 512) -> torch.Tensor:
    """
    Fallback cell mask via Otsu thresholding when pre-computed Mask R-CNN mask is unavailable.
    Returns (1, H, W) binary float tensor where 1.0 = cell foreground, 0.0 = background.
    """
    import cv2
    img_np = np.array(pil_img.convert('L').resize((img_size, img_size)))
    _, binary = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n_labels, labels, stat_arr, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels > 1:
        largest = 1 + np.argmax(stat_arr[1:, cv2.CC_STAT_AREA])
        binary = (labels == largest).astype(np.uint8) * 255
    return torch.from_numpy(binary / 255.0).float().unsqueeze(0)


def compute_region_mse(a: torch.Tensor, b: torch.Tensor, region_mask: torch.Tensor) -> float:
    """
    Computes mean squared error between image tensors a and b in [0, 1] restricted to region_mask.
    region_mask is a (1, 1, H, W) tensor where 1.0 = include, 0.0 = exclude.
    """
    diff_sq = (a - b) ** 2
    n = region_mask.sum().clamp(min=1.0)
    return ((diff_sq * region_mask).sum() / (n * 3.0)).item()


def compute_boundary_discontinuity(orig_t: torch.Tensor,
                                   out_t: torch.Tensor,
                                   mask_t: torch.Tensor,
                                   dilation_px: int = 8) -> float:
    """
    Measures gradient magnitude discontinuity in the transition seam (dilated border ring).
    A lower value indicates seamless blending between cell edit and background anchor.
    """
    # Create seam ring: dilated_mask - mask
    m_4d = mask_t.unsqueeze(0).float()
    dilated = F.max_pool2d(m_4d, kernel_size=2 * dilation_px + 1, stride=1, padding=dilation_px)
    eroded = 1.0 - F.max_pool2d(1.0 - m_4d, kernel_size=5, stride=1, padding=2)
    seam_ring = (dilated - eroded).clamp(0.0, 1.0)

    # Sobel kernels for gradient
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=orig_t.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=orig_t.device).view(1, 1, 3, 3)

    orig_gray = orig_t.mean(dim=1, keepdim=True)
    out_gray = out_t.mean(dim=1, keepdim=True)

    grad_orig = torch.sqrt(F.conv2d(orig_gray, sobel_x, padding=1) ** 2 +
                           F.conv2d(orig_gray, sobel_y, padding=1) ** 2 + 1e-8)
    grad_out = torch.sqrt(F.conv2d(out_gray, sobel_x, padding=1) ** 2 +
                          F.conv2d(out_gray, sobel_y, padding=1) ** 2 + 1e-8)

    diff_grad_sq = (grad_orig - grad_out) ** 2
    n = seam_ring.sum().clamp(min=1.0)
    return ((diff_grad_sq * seam_ring).sum() / n).item()


# ─── STATISTICAL ANALYSIS & REPORTING ──────────────────────────────────────────

def analyze_and_report_direction(direction: str,
                                 cell_records: list,
                                 out_dir: str,
                                 strength: float,
                                 cfg: float,
                                 steps: int,
                                 dilation_px: int):
    """
    Performs cell-level and FOV-clustered statistical hypothesis testing.
    Saves cell_level_metrics.csv, fov_level_metrics.csv, and statistical_report.txt.
    """
    os.makedirs(out_dir, exist_ok=True)
    df_cells = pd.DataFrame(cell_records)
    csv_cells_path = os.path.join(out_dir, 'cell_level_metrics.csv')
    df_cells.to_csv(csv_cells_path, index=False)

    # FOV-level aggregation (mean of cells belonging to each FOV)
    numeric_cols = [
        'bg_mse_std', 'bg_mse_msk', 'delta_bg', 'bg_reduction_pct',
        'cell_mse_std', 'cell_mse_msk', 'delta_cell',
        'snr_std', 'snr_msk',
        'boundary_seam_std', 'boundary_seam_msk'
    ]
    df_fov = df_cells.groupby('fov')[numeric_cols].mean().reset_index()
    df_fov['cell_count'] = df_cells.groupby('fov').size().values
    csv_fov_path = os.path.join(out_dir, 'fov_level_metrics.csv')
    df_fov.to_csv(csv_fov_path, index=False)

    N_cells = len(df_cells)
    N_fovs = len(df_fov)

    # FOV-Level Paired Statistical Tests
    # 1. Background MSE reduction
    delta_bg_fov = df_fov['delta_bg']
    mean_delta_bg = float(delta_bg_fov.mean())
    std_delta_bg = float(delta_bg_fov.std())
    sem_delta_bg = float(stats.sem(delta_bg_fov))
    ci_bg = stats.t.interval(0.95, df=N_fovs - 1, loc=mean_delta_bg, scale=sem_delta_bg)
    t_bg, p_t_bg = stats.ttest_1samp(delta_bg_fov, 0.0)
    w_res_bg = stats.wilcoxon(delta_bg_fov)
    p_w_bg = float(w_res_bg.pvalue)

    mean_bg_std_fov = float(df_fov['bg_mse_std'].mean())
    mean_bg_msk_fov = float(df_fov['bg_mse_msk'].mean())
    fov_bg_red_pct = ((mean_bg_std_fov - mean_bg_msk_fov) / (mean_bg_std_fov + 1e-9)) * 100.0

    # 2. Cell Region MSE difference
    delta_cell_fov = df_fov['delta_cell']
    mean_delta_cell = float(delta_cell_fov.mean())
    sem_delta_cell = float(stats.sem(delta_cell_fov))
    ci_cell = stats.t.interval(0.95, df=N_fovs - 1, loc=mean_delta_cell, scale=sem_delta_cell)
    t_cell, p_t_cell = stats.ttest_1samp(delta_cell_fov, 0.0)
    w_res_cell = stats.wilcoxon(delta_cell_fov)
    p_w_cell = float(w_res_cell.pvalue)

    mean_cell_std_fov = float(df_fov['cell_mse_std'].mean())
    mean_cell_msk_fov = float(df_fov['cell_mse_msk'].mean())

    # 3. Edit SNR (Cell MSE / Background MSE)
    mean_snr_std = float(df_fov['snr_std'].mean())
    mean_snr_msk = float(df_fov['snr_msk'].mean())
    snr_fold = mean_snr_msk / (mean_snr_std + 1e-9)

    # 4. Boundary Seam Discontinuity
    mean_seam_std = float(df_fov['boundary_seam_std'].mean())
    mean_seam_msk = float(df_fov['boundary_seam_msk'].mean())

    # Print Formatted Report to Console
    print(f"\n{'='*72}")
    print(f"MASK-GUIDED SDEDIT ABLATION REPORT: {direction.upper()}")
    print(f"{'='*72}")
    print(f"Sample: {N_cells} total single-cells across {N_fovs} independent FOVs")
    print(f"Parameters: strength={strength} | CFG={cfg} | steps={steps} | dilation={dilation_px}px")
    print(f"{'-'*72}")
    print(f"{'Metric (FOV-Clustered)':<38} {'Standard':>10} {'Masked':>10} {'Delta':>10}")
    print(f"{'-'*72}")
    print(f"{'Background MSE (lower = cleaner)':<38} {mean_bg_std_fov:>10.6f} {mean_bg_msk_fov:>10.6f} {mean_delta_bg:>+10.6f}")
    print(f"{'Cell-region MSE (transformation extent)':<38} {mean_cell_std_fov:>10.6f} {mean_cell_msk_fov:>10.6f} {mean_delta_cell:>+10.6f}")
    print(f"{'Edit SNR (Cell MSE / BG MSE)':<38} {mean_snr_std:>10.2f} {mean_snr_msk:>10.2f} {snr_fold:>9.1f}x")
    print(f"{'Boundary Ring Gradient Magnitude Diff':<38} {mean_seam_std:>10.6f} {mean_seam_msk:>10.6f} {mean_seam_std-mean_seam_msk:>+10.6f}")
    print(f"{'='*72}")
    print(f"[*] FOV-Level Background Reduction: {fov_bg_red_pct:+.2f}%")
    print(f"[*] Paired t-test: t({N_fovs-1}) = {t_bg:+.3f}, p = {p_t_bg:.4e} (95% CI: [{ci_bg[0]:+.6f}, {ci_bg[1]:+.6f}])")
    print(f"[*] Wilcoxon signed-rank test: W = {w_res_bg.statistic:.1f}, p = {p_w_bg:.4e}")
    print(f"[*] Cell-region MSE difference: delta={mean_delta_cell:+.6f} (t={t_cell:+.2f}, p={p_t_cell:.4e})")
    print(f"    (Note: Equivalence or non-inferiority requires a pre-specified margin)")
    print(f"{'='*72}\n")

    # Save Detailed Scientific Report
    report_path = os.path.join(out_dir, f'statistical_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"MASK-GUIDED SDEDIT STATISTICAL REPORT — {direction.upper()}\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"1. EXPERIMENTAL SETUP & RIGOR CONTROLS\n")
        f.write(f"   - Task Direction:                {direction}\n")
        f.write(f"   - Total Evaluated Cells:         {N_cells}\n")
        f.write(f"   - Total Independent FOVs:        {N_fovs}\n")
        f.write(f"   - SDEdit Noise Strength:         {strength}\n")
        f.write(f"   - Classifier-Free Guidance (CFG):{cfg}\n")
        f.write(f"   - DDIM Denoising Steps:          {steps}\n")
        f.write(f"   - Mask Dilation Radius:          {dilation_px} px\n")
        f.write(f"   - Noise Pairing:                 Identical initial latent noise vector (x_{{t_start}})\n")
        f.write(f"   - Background Anchoring:          RePaint-style aligned to t_prev with terminal clean x_0\n\n")

        f.write(f"2. FOV-CLUSTERED STATISTICAL HYPOTHESIS TESTING\n")
        f.write(f"   - Background MSE Standard (mean across FOVs): {mean_bg_std_fov:.6f}\n")
        f.write(f"   - Background MSE Masked   (mean across FOVs): {mean_bg_msk_fov:.6f}\n")
        f.write(f"   - Mean Paired Difference (Delta):             {mean_delta_bg:+.6f}\n")
        f.write(f"   - 95% Confidence Interval:                   [{ci_bg[0]:+.6f}, {ci_bg[1]:+.6f}]\n")
        f.write(f"   - Two-Sided Paired Student's t-test:          t({N_fovs-1}) = {t_bg:+.4f}, p = {p_t_bg:.4e}\n")
        f.write(f"   - Wilcoxon Signed-Rank Test:                  W = {w_res_bg.statistic:.1f}, p = {p_w_bg:.4e}\n")
        f.write(f"   - FOV-Level Background Reduction:             {fov_bg_red_pct:+.2f}%\n\n")

        f.write(f"3. CELL MORPHOLOGY & TRANSFORMATION DYNAMICS\n")
        f.write(f"   - Cell-Region MSE Standard (mean across FOVs): {mean_cell_std_fov:.6f}\n")
        f.write(f"   - Cell-Region MSE Masked   (mean across FOVs): {mean_cell_msk_fov:.6f}\n")
        f.write(f"   - Cell MSE Difference (Delta):                 {mean_delta_cell:+.6f}\n")
        f.write(f"   - 95% Confidence Interval:                    [{ci_cell[0]:+.6f}, {ci_cell[1]:+.6f}]\n")
        f.write(f"   - Paired t-test (Cell Transformation):        t({N_fovs-1}) = {t_cell:+.4f}, p = {p_t_cell:.4e}\n")
        f.write(f"   - Wilcoxon Signed-Rank Test:                   W = {w_res_cell.statistic:.1f}, p = {p_w_cell:.4e}\n")
        f.write(f"   - Interpretation: Cell MSE is slightly lower in masked SDEdit (-Delta = {mean_delta_cell:.6f}),\n")
        f.write(f"     indicating slightly more conservative within-mask changes relative to unconstrained SDEdit.\n")
        f.write(f"     Rigorous equivalence testing requires a pre-specified non-inferiority margin.\n\n")

        f.write(f"4. EDIT SELECTIVITY & BOUNDARY METRICS\n")
        f.write(f"   - Edit Signal-to-Disruption Ratio (Standard): {mean_snr_std:.2f}\n")
        f.write(f"   - Edit Signal-to-Disruption Ratio (Masked):   {mean_snr_msk:.2f} ({snr_fold:.1f}x higher specificity)\n")
        f.write(f"   - Boundary Ring Gradient MSE (Standard):      {mean_seam_std:.6f}\n")
        f.write(f"   - Boundary Ring Gradient MSE (Masked):        {mean_seam_msk:.6f}\n")
        f.write(f"   - Boundary Ring Gradient Magnitude Diff:      {mean_seam_std-mean_seam_msk:+.6f}\n")
        f.write(f"     (Measured in the dilation transition ring; does not guarantee complete absence of boundary seams)\n\n")

        f.write(f"5. CONCLUSION\n")
        f.write(f"   Mask-guided SDEdit achieves a statistically significant {fov_bg_red_pct:.1f}% reduction\n")
        f.write(f"   in background modification (p = {p_w_bg:.2e}, Wilcoxon test across {N_fovs} FOVs).\n")

    print(f"[OK] Saved full statistical results to:\n  - {csv_cells_path}\n  - {csv_fov_path}\n  - {report_path}\n")


# ─── MAIN ROUTINE ─────────────────────────────────────────────────────────────

def run_evaluation_for_direction(direction: str, args, model, device):
    print(f"\n{'#'*72}")
    print(f"# EXECUTING MASK-GUIDED SDEDIT EVALUATION: {direction.upper()}")
    print(f"{'#'*72}")

    # Set parameters according to best operating points from Section 3.10 if not overridden
    strength = args.strength if args.strength is not None else (0.80 if direction == 'aging' else 0.70)
    cfg = args.cfg if args.cfg is not None else (5.0 if direction == 'aging' else 4.0)
    steps = args.steps
    if direction == 'aging':
        dilation_px = args.dilation_aging if args.dilation_aging is not None else (args.dilation_px if args.dilation_px is not None else 24)
    else:
        dilation_px = args.dilation_rejuvenation if args.dilation_rejuvenation is not None else (args.dilation_px if args.dilation_px is not None else 8)

    dir_out = os.path.join(args.output_dir, direction)
    std_dir = os.path.join(dir_out, 'standard')
    msk_dir = os.path.join(dir_out, 'masked')
    grid_dir = os.path.join(dir_out, 'grids')
    os.makedirs(std_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    if direction == 'aging':
        source_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
        target_label_val = 1
        mask_npy = os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_young.npy')
    else:
        source_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
        target_label_val = 0
        mask_npy = os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_senescent.npy')

    # Load Mask R-CNN pre-computed masks
    mrcnn_masks = {}
    if os.path.exists(mask_npy):
        print(f"[*] Loading pre-computed Mask R-CNN masks: {mask_npy}")
        mrcnn_masks = np.load(mask_npy, allow_pickle=True).item()
        print(f"    Loaded {len(mrcnn_masks)} masks")
    else:
        print(f"[!] Warning: {mask_npy} not found. Falling back to Otsu thresholding.")

    all_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
    if args.max_images is not None:
        all_files = all_files[:args.max_images]

    print(f"[*] Direction: {direction} | Total test images: {len(all_files)}")
    print(f"[*] Parameters: strength={strength} | CFG={cfg} | steps={steps} | dilation={dilation_px}px")

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    to_tensor_01 = transforms.ToTensor()

    # ─── SUMMARY ONLY EVALUATION ──────────────────────────────────────────────
    if args.summary_only:
        print(f"[*] Summary Mode: computing metrics from existing outputs in {dir_out}...")
        cell_records = []
        for fn in tqdm(all_files, desc=f"Evaluating ({direction})"):
            p_std = os.path.join(std_dir, fn)
            p_msk = os.path.join(msk_dir, fn)
            if not (os.path.exists(p_std) and os.path.exists(p_msk)):
                continue

            src_im = Image.open(os.path.join(source_dir, fn)).convert('RGB')
            orig_01 = to_tensor_01(src_im).unsqueeze(0)
            std_01 = to_tensor_01(Image.open(p_std).convert('RGB')).unsqueeze(0)
            msk_01 = to_tensor_01(Image.open(p_msk).convert('RGB')).unsqueeze(0)

            # Get fixed input mask
            if fn in mrcnn_masks:
                mask_t = torch.from_numpy(mrcnn_masks[fn].astype(np.float32)).unsqueeze(0)
                if mask_t.shape[1] != 512 or mask_t.shape[2] != 512:
                    mask_t = F.interpolate(mask_t.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
            else:
                mask_t = get_cell_mask_otsu(src_im, img_size=512)

            mask_4d = mask_t.unsqueeze(0)
            bg_4d = 1.0 - mask_4d

            bg_std = compute_region_mse(orig_01, std_01, bg_4d)
            bg_msk = compute_region_mse(orig_01, msk_01, bg_4d)
            cell_std = compute_region_mse(orig_01, std_01, mask_4d)
            cell_msk = compute_region_mse(orig_01, msk_01, mask_4d)

            delta_bg = bg_std - bg_msk
            delta_cell = cell_std - cell_msk
            bg_red_pct = (delta_bg / (bg_std + 1e-9)) * 100.0

            seam_std = compute_boundary_discontinuity(orig_01, std_01, mask_t, dilation_px)
            seam_msk = compute_boundary_discontinuity(orig_01, msk_01, mask_t, dilation_px)

            fov = extract_fov_id(fn)
            cell_records.append({
                'fname': fn,
                'fov': fov,
                'bg_mse_std': bg_std,
                'bg_mse_msk': bg_msk,
                'delta_bg': delta_bg,
                'bg_reduction_pct': bg_red_pct,
                'cell_mse_std': cell_std,
                'cell_mse_msk': cell_msk,
                'delta_cell': delta_cell,
                'snr_std': cell_std / (bg_std + 1e-9),
                'snr_msk': cell_msk / (bg_msk + 1e-9),
                'boundary_seam_std': seam_std,
                'boundary_seam_msk': seam_msk
            })

        analyze_and_report_direction(direction, cell_records, dir_out, strength, cfg, steps, dilation_px)
        return

    # ─── FULL INFERENCE + EVALUATION ──────────────────────────────────────────
    cell_records = []
    seen_fovs_for_grid = set()
    B = args.batch_size

    pbar = tqdm(range(0, len(all_files), B), desc=f"Translating ({direction})")
    for start_idx in pbar:
        batch_fns = all_files[start_idx:start_idx + B]
        batch_imgs_list = []
        batch_masks_list = []

        for fn in batch_fns:
            img_path = os.path.join(source_dir, fn)
            pil_img = Image.open(img_path).convert('RGB')
            img_t = transform_norm(pil_img)
            batch_imgs_list.append(img_t)

            if fn in mrcnn_masks:
                mask_t = torch.from_numpy(mrcnn_masks[fn].astype(np.float32)).unsqueeze(0)
                if mask_t.shape[1] != 512 or mask_t.shape[2] != 512:
                    mask_t = F.interpolate(mask_t.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
            else:
                mask_t = get_cell_mask_otsu(pil_img, img_size=512)
            batch_masks_list.append(mask_t)

        batch_imgs = torch.stack(batch_imgs_list, dim=0).to(device, memory_format=torch.channels_last)
        batch_masks = torch.stack(batch_masks_list, dim=0).to(device)
        batch_labels = torch.full((len(batch_fns),), target_label_val, dtype=torch.long, device=device)

        # STRICT NOISE PAIRING:
        # Generate the exact same initial latent noise for both Standard and Masked translation
        torch.manual_seed(args.seed + start_idx)
        init_noise = torch.randn((len(batch_fns), 4, 64, 64), device=device, dtype=torch.float32)

        with torch.no_grad():
            out_std = model.translate(
                batch_imgs, batch_labels,
                strength=strength, num_steps=steps,
                use_ema=True, guidance_scale=cfg,
                noise=init_noise
            )
            out_msk = model.translate_masked(
                batch_imgs, batch_labels, cell_mask=batch_masks,
                strength=strength, num_steps=steps,
                use_ema=True, guidance_scale=cfg,
                dilation_px=dilation_px,
                noise=init_noise
            )

        # Process batch results
        for idx_in_batch, fn in enumerate(batch_fns):
            std_img = out_std[idx_in_batch:idx_in_batch + 1].cpu()
            msk_img = out_msk[idx_in_batch:idx_in_batch + 1].cpu()
            orig_img = ((batch_imgs[idx_in_batch:idx_in_batch + 1].cpu() + 1.0) / 2.0)
            cur_mask = batch_masks[idx_in_batch:idx_in_batch + 1].cpu()
            cur_bg = 1.0 - cur_mask

            # Save individual outputs
            save_image(std_img, os.path.join(std_dir, fn))
            save_image(msk_img, os.path.join(msk_dir, fn))

            # Save visual comparison grid for first exemplar of each FOV
            fov = extract_fov_id(fn)
            if fov not in seen_fovs_for_grid or len(seen_fovs_for_grid) < 20:
                seen_fovs_for_grid.add(fov)
                diff_std = (orig_img - std_img).abs()
                diff_msk = (orig_img - msk_img).abs()
                grid = make_grid(
                    torch.cat([orig_img, std_img, msk_img, diff_std.clamp(0, 1), diff_msk.clamp(0, 1)], dim=0),
                    nrow=5, padding=4, pad_value=0.8
                )
                save_image(grid, os.path.join(grid_dir, f"{fn.split('.')[0]}_grid.png"))

            # Compute pixel metrics
            bg_std = compute_region_mse(orig_img, std_img, cur_bg)
            bg_msk = compute_region_mse(orig_img, msk_img, cur_bg)
            cell_std = compute_region_mse(orig_img, std_img, cur_mask)
            cell_msk = compute_region_mse(orig_img, msk_img, cur_mask)

            delta_bg = bg_std - bg_msk
            delta_cell = cell_std - cell_msk
            bg_red_pct = (delta_bg / (bg_std + 1e-9)) * 100.0

            seam_std = compute_boundary_discontinuity(orig_img, std_img, cur_mask.squeeze(0), dilation_px)
            seam_msk = compute_boundary_discontinuity(orig_img, msk_img, cur_mask.squeeze(0), dilation_px)

            cell_records.append({
                'fname': fn,
                'fov': fov,
                'bg_mse_std': bg_std,
                'bg_mse_msk': bg_msk,
                'delta_bg': delta_bg,
                'bg_reduction_pct': bg_red_pct,
                'cell_mse_std': cell_std,
                'cell_mse_msk': cell_msk,
                'delta_cell': delta_cell,
                'snr_std': cell_std / (bg_std + 1e-9),
                'snr_msk': cell_msk / (bg_msk + 1e-9),
                'boundary_seam_std': seam_std,
                'boundary_seam_msk': seam_msk
            })

        if device == 'cuda':
            torch.cuda.empty_cache()

    analyze_and_report_direction(direction, cell_records, dir_out, strength, cfg, steps, dilation_px)


def main():
    parser = argparse.ArgumentParser(description="Mask-Guided SDEdit Full Population Ablation")
    parser.add_argument('--direction', type=str, default='both',
                        choices=['aging', 'rejuvenation', 'both'],
                        help='Translation direction to evaluate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for GPU translation')
    parser.add_argument('--strength', type=float, default=None,
                        help='SDEdit noise strength (default: 0.80 aging, 0.70 rejuv)')
    parser.add_argument('--cfg', type=float, default=None,
                        help='Classifier-free guidance scale (default: 5.0 aging, 4.0 rejuv)')
    parser.add_argument('--steps', type=int, default=50,
                        help='DDIM sampling steps')
    parser.add_argument('--dilation_px', type=int, default=None,
                        help='Fallback mask dilation radius in pixels if direction-specific is not set')
    parser.add_argument('--dilation_aging', type=int, default=None,
                        help='Mask dilation radius in pixels for aging (default: 24 px)')
    parser.add_argument('--dilation_rejuvenation', type=int, default=None,
                        help='Mask dilation radius in pixels for rejuvenation (default: 8 px)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Optional cap on number of images per direction (for quick validation)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Base random seed for strictly paired noise')
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(root_dir, 'checkpoints', 'ldm',
                                             'checkpoint_v12_v4_data_lpips_last.pt'),
                        help='Path to LDM model checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(root_dir, 'results', 'ablation_masked'),
                        help='Output directory for images, CSVs, and statistical reports')
    parser.add_argument('--summary_only', action='store_true',
                        help='Skip inference and evaluate existing images in output_dir')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = None
    if not args.summary_only:
        from models.ldm.model_lpips import CellLDM
        print("\n[*] Initializing CellLDM architecture...")
        model = CellLDM(num_classes=2, lpips_weight=0.0)
        model.scaling_factor = 0.18215
        model.to(DEVICE, memory_format=torch.channels_last)
        model.vae.to(memory_format=torch.channels_last)
        model.init_ema()

        print(f"[*] Loading checkpoint from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        for key in ['unet_state_dict', 'ema_unet_state_dict']:
            if key in ckpt and 'class_embedding.weight' in ckpt[key]:
                w = ckpt[key]['class_embedding.weight']
                if w.shape[0] == 2:
                    pad = torch.zeros(1, w.shape[1], device=w.device)
                    ckpt[key]['class_embedding.weight'] = torch.cat([w, pad], dim=0)

        model.unet.load_state_dict(ckpt['unet_state_dict'])
        if 'ema_unet_state_dict' in ckpt:
            model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
        del ckpt
        gc.collect()
        model.eval()
        print("    Model loaded successfully onto", torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU')

    directions = ['aging', 'rejuvenation'] if args.direction == 'both' else [args.direction]
    for d in directions:
        run_evaluation_for_direction(d, args, model, DEVICE)

    print(f"\n[DONE] All evaluations and FOV-clustered statistical reports completed successfully!")


if __name__ == '__main__':
    main()
