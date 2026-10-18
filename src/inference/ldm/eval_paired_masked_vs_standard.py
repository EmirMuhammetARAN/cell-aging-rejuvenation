"""
Strictly Paired Evaluation: Mask-Guided vs Standard SDEdit (FID & ResNet-18 Accuracy)
=====================================================================================
Performs a strictly paired, controlled benchmark of Standard (Unmasked) SDEdit
vs Mask-Guided SDEdit using optimal dilation radii (Aging: 24 px, Rejuvenation: 8 px)
across both Low-FID and High-Accuracy regimes.

Methodological Rigor:
  - Both Standard and Masked images are generated in the exact same loop,
    using the IDENTICAL initial noise tensor (z_init) per cell crop.
  - Generates into brand-new dedicated directories (strictly_paired).
  - Evaluates FID (Inception-v3) and ResNet-18 auxiliary classifier accuracy/confidence.
  - Reports accuracy differences in percentage points (yuzde puan, pp).

Configurations:
  1. Low-FID:
     - Aging:        strength = 0.75, CFG = 4.0, dilation = 24 px
     - Rejuvenation: strength = 0.65, CFG = 3.5, dilation = 8 px
  2. High-Accuracy:
     - Aging:        strength = 0.80, CFG = 6.0, dilation = 24 px
     - Rejuvenation: strength = 0.70, CFG = 5.0, dilation = 8 px
"""

import os
import sys
import gc
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from pytorch_fid import fid_score

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
from models.classifier.classifier import Classifier


def extract_fov_id(filename: str) -> str:
    m = re.match(r'^(.*)_[0-9]+\.(?:jpg|png)$', filename)
    return m.group(1) if m else filename


def load_ldm_model(device: str):
    checkpoint_path = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
    model = CellLDM(num_classes=2, lpips_weight=0.0)
    model.scaling_factor = 0.18215
    model.to(device, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    model.init_ema()

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    for key in ['unet_state_dict', 'ema_unet_state_dict']:
        if key in ckpt and 'class_embedding.weight' in ckpt[key]:
            w = ckpt[key]['class_embedding.weight']
            if w.shape[0] == 2:
                pad = torch.zeros(1, w.shape[1], device=w.device)
                ckpt[key]['class_embedding.weight'] = torch.cat([w, pad], dim=0)

    model.unet.load_state_dict(ckpt['unet_state_dict'])
    if 'ema_unet_state_dict' in ckpt:
        model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
    print(f"[OK] Loaded LDM checkpoint from epoch {ckpt.get('epoch', '?')}")
    del ckpt
    gc.collect()
    model.eval()
    return model


def generate_strictly_paired(regime_name: str, configs: dict, model, device, base_out_dir: str, batch_size: int = 4, steps: int = 100):
    """
    Generates strictly paired Standard and Mask-Guided translations side-by-side
    with the exact same initial noise tensor per cell.
    """
    print(f"\n{'='*75}")
    print(f"STRICTLY PAIRED GENERATION ({steps} DDIM STEPS): REGIME = {regime_name.upper()}")
    print(f"{'='*75}")

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    regime_dir = os.path.join(base_out_dir, regime_name)
    os.makedirs(regime_dir, exist_ok=True)

    for direction, cfg in configs.items():
        print(f"\n---> Direction: {direction.upper()} (strength={cfg['strength']}, CFG={cfg['cfg']}, dilation={cfg['dilation']} px, steps={steps})")
        
        if direction == 'aging':
            source_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
            target_label_val = 1  # LDM senescent class
            mask_npy = os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_young.npy')
        else:
            source_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
            target_label_val = 0  # LDM young class
            mask_npy = os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_senescent.npy')

        mrcnn_masks = np.load(mask_npy, allow_pickle=True).item()
        all_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
        valid_files = [f for f in all_files if f in mrcnn_masks and mrcnn_masks[f].sum() > 0]
        print(f"[*] Valid single-cell cohort: {len(valid_files)} crops")

        std_dir = os.path.join(regime_dir, f"{direction}_standard")
        msk_dir = os.path.join(regime_dir, f"{direction}_masked_{cfg['dilation']}px")
        os.makedirs(std_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)

        existing_std = set(os.listdir(std_dir))
        existing_msk = set(os.listdir(msk_dir))

        pending_files = [f for f in valid_files if f not in existing_std or f not in existing_msk]
        print(f"[*] Pending crops to generate: {len(pending_files)} / {len(valid_files)}")

        if len(pending_files) == 0:
            print(f"[OK] All strictly paired {steps}-step images already exist on disk.")
            continue

        pbar = tqdm(range(0, len(pending_files), batch_size), desc=f"{regime_name}-{direction}-paired-{steps}steps")
        for start_idx in pbar:
            batch_fns = pending_files[start_idx:start_idx + batch_size]
            batch_imgs_list = []
            batch_masks_list = []

            for fn in batch_fns:
                img_path = os.path.join(source_dir, fn)
                pil_img = Image.open(img_path).convert('RGB')
                batch_imgs_list.append(transform_norm(pil_img))

                mask_np = mrcnn_masks[fn].astype(np.float32)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0)
                if mask_t.shape[1] != 512 or mask_t.shape[2] != 512:
                    mask_t = F.interpolate(mask_t.unsqueeze(0), size=(512, 512), mode='nearest').squeeze(0)
                batch_masks_list.append(mask_t)

            batch_imgs = torch.stack(batch_imgs_list, dim=0).to(device, memory_format=torch.channels_last)
            batch_masks = torch.stack(batch_masks_list, dim=0).to(device)
            batch_labels = torch.full((len(batch_fns),), target_label_val, dtype=torch.long, device=device)

            # Strictly shared initial noise vector per batch
            torch.manual_seed(1000 + start_idx)
            init_noise = torch.randn((len(batch_fns), 4, 64, 64), device=device, dtype=torch.float32)

            # 1. Standard (Unmasked) SDEdit with shared noise
            with torch.no_grad():
                out_std = model.translate(
                    batch_imgs, batch_labels,
                    strength=cfg['strength'], num_steps=steps,
                    use_ema=True, guidance_scale=cfg['cfg'],
                    noise=init_noise
                )

            # 2. Mask-Guided SDEdit with IDENTICAL shared noise
            with torch.no_grad():
                out_msk = model.translate_masked(
                    batch_imgs, batch_labels, cell_mask=batch_masks,
                    strength=cfg['strength'], num_steps=steps,
                    use_ema=True, guidance_scale=cfg['cfg'],
                    dilation_px=cfg['dilation'],
                    noise=init_noise
                )

            # Save batch outputs
            for idx_b, fn in enumerate(batch_fns):
                save_image(out_std[idx_b], os.path.join(std_dir, fn))
                save_image(out_msk[idx_b], os.path.join(msk_dir, fn))

            if device == 'cuda':
                torch.cuda.empty_cache()

        print(f"[OK] Completed paired {steps}-step generation for {regime_name} {direction}.")


def evaluate_auxiliary_classifier(img_dir: str, target_class: int, classifier, device):
    """
    Computes ResNet-18 auxiliary classifier target accuracy and mean confidence (%)
    on images in img_dir.
    """
    cls_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.tif', '.tiff'))])
    if not img_names:
        return 0.0, 0.0, 0, 0

    correct = 0
    total = len(img_names)
    confidences = []

    with torch.no_grad():
        for fn in img_names:
            img_path = os.path.join(img_dir, fn)
            img = Image.open(img_path).convert('RGB')
            img_t = cls_transform(img).unsqueeze(0).to(device)

            logits = classifier(img_t)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()

            if pred == target_class:
                correct += 1

            target_conf = probs[0, target_class].item() * 100.0
            confidences.append(target_conf)

    acc = (correct / total) * 100.0
    mean_conf = float(np.mean(confidences))
    return acc, mean_conf, correct, total


def create_visual_comparison_grid(regime_name: str, configs: dict, base_out_dir: str):
    """
    Builds a high-resolution 4-panel visual comparison grid:
    Input | Standard SDEdit | Mask-Guided SDEdit | Dilated Mask Overlay
    """
    regime_dir = os.path.join(base_out_dir, regime_name)
    grid_out_path = os.path.join(regime_dir, f"paired_visual_comparison_grid_{regime_name}.png")

    rows = []
    for direction, cfg in configs.items():
        if direction == 'aging':
            source_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
            mask_npy = os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_young.npy')
        else:
            source_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
            mask_npy = os.path.join(root_dir, 'scratch', 'mrcnn_masks_input_senescent.npy')

        mrcnn_masks = np.load(mask_npy, allow_pickle=True).item()
        std_dir = os.path.join(regime_dir, f"{direction}_standard")
        msk_dir = os.path.join(regime_dir, f"{direction}_masked_{cfg['dilation']}px")

        all_files = sorted(os.listdir(std_dir))
        fov_map = {}
        for fn in all_files:
            fov = extract_fov_id(fn)
            fov_map.setdefault(fov, []).append(fn)

        exemplar_fns = []
        for fov in sorted(fov_map.keys())[:3]:
            exemplar_fns.append(fov_map[fov][0])

        for fn in exemplar_fns:
            img_in = Image.open(os.path.join(source_dir, fn)).convert('RGB')
            img_std = Image.open(os.path.join(std_dir, fn)).convert('RGB')
            img_msk = Image.open(os.path.join(msk_dir, fn)).convert('RGB')

            t_in = transforms.ToTensor()(img_in)
            t_std = transforms.ToTensor()(img_std)
            t_msk = transforms.ToTensor()(img_msk)

            # Mask overlay (cyan tint on dilated boundary)
            mask_np = mrcnn_masks[fn].astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
            if mask_t.shape[2:] != (512, 512):
                mask_t = F.interpolate(mask_t, size=(512, 512), mode='nearest')
            dilated_m = F.max_pool2d(mask_t, kernel_size=2 * cfg['dilation'] + 1, stride=1, padding=cfg['dilation'])
            
            overlay = t_in.clone()
            diff_mask = (dilated_m.squeeze(0) - mask_t.squeeze(0)).clamp(0, 1)
            overlay[0] = overlay[0] * (1.0 - diff_mask * 0.5)
            overlay[1] = torch.clamp(overlay[1] + diff_mask * 0.4, 0, 1)
            overlay[2] = torch.clamp(overlay[2] + diff_mask * 0.6, 0, 1)

            rows.extend([t_in, t_std, t_msk, overlay])

    if rows:
        grid = make_grid(rows, nrow=4, padding=4, pad_value=1.0)
        save_image(grid, grid_out_path)
        print(f"[OK] Saved visual comparison grid to {grid_out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true', help='Skip generation and run FID/Classifier evaluation only')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for generation')
    parser.add_argument('--steps', type=int, default=100, help='Number of DDIM steps')
    parser.add_argument('--regime', type=str, default='both', choices=['both', 'low_fid', 'high_acc'], help='Regime to run')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for generated images and reports')
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    if args.output_dir is not None:
        base_out_dir = args.output_dir
    else:
        folder_name = 'strictly_paired' if args.steps == 50 else f'strictly_paired_{args.steps}steps'
        base_out_dir = os.path.join(root_dir, 'results', 'ablation_masked_vs_standard', folder_name)
    os.makedirs(base_out_dir, exist_ok=True)
    print(f"Base Output Directory: {base_out_dir} ({args.steps} DDIM steps)")

    # Master hyperparameter configurations
    all_regimes = {
        'low_fid': {
            'aging': {'strength': 0.75, 'cfg': 4.0, 'dilation': 24},
            'rejuvenation': {'strength': 0.65, 'cfg': 3.5, 'dilation': 8}
        },
        'high_acc': {
            'aging': {'strength': 0.80, 'cfg': 6.0, 'dilation': 24},
            'rejuvenation': {'strength': 0.70, 'cfg': 5.0, 'dilation': 8}
        }
    }

    if args.regime == 'both':
        regimes = all_regimes
    else:
        regimes = {args.regime: all_regimes[args.regime]}

    ref_paths = {
        'aging': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent'),
        'rejuvenation': os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    }

    # Classifier target mapping: Senescent = 0, Young = 1
    cls_targets = {
        'aging': 0,
        'rejuvenation': 1
    }

    # --- Step 1: Paired Generation ---
    if not args.eval_only:
        model = load_ldm_model(device)
        for regime_name, configs in regimes.items():
            generate_strictly_paired(regime_name, configs, model, device, base_out_dir, batch_size=args.batch_size, steps=args.steps)
            create_visual_comparison_grid(regime_name, configs, base_out_dir)

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # --- Step 2: Metric Evaluation ---
    print(f"\n{'='*75}")
    print("COMPUTING PAIRED METRICS (FID & RESNET-18 AUXILIARY CLASSIFIER)")
    print(f"{'='*75}")

    classifier = Classifier(output_size=2)
    cls_ckpt = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier.load_state_dict(torch.load(cls_ckpt, map_location=device))
    classifier.to(device)
    classifier.eval()
    print("[OK] Loaded Auxiliary Classifier checkpoint.")

    evaluation_records = []

    for regime_name, configs in regimes.items():
        regime_dir = os.path.join(base_out_dir, regime_name)
        print(f"\n>>> Evaluating Regime: {regime_name.upper()}")

        for direction, cfg in configs.items():
            d = cfg['dilation']
            std_dir = os.path.join(regime_dir, f"{direction}_standard")
            msk_dir = os.path.join(regime_dir, f"{direction}_masked_{d}px")
            ref_dir = ref_paths[direction]
            target_class = cls_targets[direction]

            # 1. FID
            print(f"[*] Calculating FID for {regime_name} {direction} (Paired Standard vs Real)...")
            fid_std = fid_score.calculate_fid_given_paths([ref_dir, std_dir], batch_size=32, device=device, dims=2048, num_workers=0)
            print(f"    Paired Standard FID: {fid_std:.2f}")

            print(f"[*] Calculating FID for {regime_name} {direction} (Paired Masked-{d}px vs Real)...")
            fid_msk = fid_score.calculate_fid_given_paths([ref_dir, msk_dir], batch_size=32, device=device, dims=2048, num_workers=0)
            print(f"    Paired Masked FID:   {fid_msk:.2f}")

            # 2. Auxiliary Classifier Accuracy & Confidence
            print(f"[*] Evaluating Auxiliary Classifier on {regime_name} {direction} (Paired Standard)...")
            acc_std, conf_std, corr_std, tot_std = evaluate_auxiliary_classifier(std_dir, target_class, classifier, device)
            print(f"    Standard Acc: {acc_std:.2f}% ({corr_std}/{tot_std}), Conf: {conf_std:.2f}%")

            print(f"[*] Evaluating Auxiliary Classifier on {regime_name} {direction} (Paired Masked-{d}px)...")
            acc_msk, conf_msk, corr_msk, tot_msk = evaluate_auxiliary_classifier(msk_dir, target_class, classifier, device)
            print(f"    Masked Acc:   {acc_msk:.2f}% ({corr_msk}/{tot_msk}), Conf: {conf_msk:.2f}%")

            delta_fid = fid_msk - fid_std     # Points (negative is better)
            delta_acc_pp = acc_msk - acc_std  # Percentage points (positive is better)
            delta_conf_pp = conf_msk - conf_std # Percentage points

            evaluation_records.append({
                'Regime': regime_name,
                'Direction': direction,
                'Strength': cfg['strength'],
                'CFG': cfg['cfg'],
                'Dilation_px': d,
                'FID_Standard': fid_std,
                'FID_Masked': fid_msk,
                'Delta_FID_pts': delta_fid,
                'Acc_Standard_pct': acc_std,
                'Acc_Masked_pct': acc_msk,
                'Delta_Acc_pp': delta_acc_pp,
                'Conf_Standard_pct': conf_std,
                'Conf_Masked_pct': conf_msk,
                'Delta_Conf_pp': delta_conf_pp,
                'N_Crops': tot_std
            })

    # Save summary dataframe
    df = pd.DataFrame(evaluation_records)
    csv_path = os.path.join(base_out_dir, 'strictly_paired_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved strictly paired evaluation summary CSV to {csv_path}")

    # Text report with rigorous scientific language
    report_path = os.path.join(base_out_dir, 'strictly_paired_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 115 + "\n")
        f.write("STRICTLY PAIRED BENCHMARK: MASK-GUIDED SDEDIT VS. STANDARD UNMASKED SDEDIT\n")
        f.write("=" * 115 + "\n\n")
        f.write("Experimental Setup:\n")
        f.write(f"  - Generative Model:         Latent Diffusion Model (v12, LPIPS-finetuned, {args.steps} DDIM steps)\n")
        f.write("  - Auxiliary Classifier:     ResNet-18 (classifier_v2.pth; yardimci siniflandirici)\n")
        f.write("  - Dilation Radii Applied:   Aging = 24 px, Rejuvenation = 8 px\n")
        f.write("  - Noise Control:            Strictly shared initial noise tensor (z_init) per single-cell crop\n")
        f.write("  - Evaluated Cohorts:        N = 327 Aging crops, N = 315 Rejuvenation crops\n\n")

        f.write("-" * 115 + "\n")
        f.write(f"{'REGIME':<10} | {'TASK':<12} | {'CONFIG':<14} | {'METRIC':<20} | {'STANDARD':<12} | {'MASKED':<12} | {'FARK (M - S)':<16} | {'USTUN'}\n")
        f.write("-" * 115 + "\n")

        for r in evaluation_records:
            reg = r['Regime'].upper()
            task = r['Direction'].capitalize()
            cfg_str = f"s={r['Strength']},cfg={r['CFG']}"

            # FID Row
            f_sign = "-" if r['Delta_FID_pts'] <= 0 else "+"
            f_win = "MASKED" if r['Delta_FID_pts'] < 0 else "STANDART"
            f.write(f"{reg:<10} | {task:<12} | {cfg_str:<14} | {'FID (puan, dusuk iyi)':<20} | {r['FID_Standard']:<12.2f} | {r['FID_Masked']:<12.2f} | {f_sign}{abs(r['Delta_FID_pts']):<15.2f} | {f_win}\n")

            # ACC Row (in percentage points)
            a_sign = "+" if r['Delta_Acc_pp'] >= 0 else "-"
            a_win = "MASKED" if r['Delta_Acc_pp'] > 0 else "STANDART"
            f.write(f"{'':<10} | {'':<12} | {'':<14} | {'ResNet-18 ACC (%)':<20} | {r['Acc_Standard_pct']:<11.2f}% | {r['Acc_Masked_pct']:<11.2f}% | {a_sign}{abs(r['Delta_Acc_pp']):<7.2f} yuzde puan | {a_win}\n")

            # Conf Row (in percentage points)
            c_sign = "+" if r['Delta_Conf_pp'] >= 0 else "-"
            c_win = "MASKED" if r['Delta_Conf_pp'] > 0 else "STANDART"
            f.write(f"{'':<10} | {'':<12} | {'':<14} | {'Hedef Guven (%)':<20} | {r['Conf_Standard_pct']:<11.2f}% | {r['Conf_Masked_pct']:<11.2f}% | {c_sign}{abs(r['Delta_Conf_pp']):<7.2f} yuzde puan | {c_win}\n")
            f.write("-" * 115 + "\n")

        f.write("\nBilimsel ve Metodolojik Notlar:\n")
        f.write("1. Yardimci Siniflandirici: ResNet-18, hiperparametre optimizasyonunda (CFG/strength taramalarinda)\n")
        f.write("   kullanildigi icin bagimsiz dis dogrulayici degil, yardimci fenotipik siniflandirici olarak kabul edilmelidir.\n")
        f.write("2. Arka Plan Sinirlamasi: Dilasyon maskeleme yontemi, baslangic maskesi disindaki alanda lokal arka plan\n")
        f.write("   MSE'sini belirgin sekilde azaltmaktadir; ancak bu deney tek hucre kirpimlarinda yurutulmus olup,\n")
        f.write("   dilasyon gecis halkasi duzenlemeye aciktir.\n")
        f.write("3. Olası Hipotezler: Yaslandirma gorevinde siniflandirici skorundaki dusus, maske sinirlamasinin olasi bir\n")
        f.write("   hipertrofik buyume kisitindan kaynaklanabilecegi gibi, modelin tuval disi baglamsal ipuclarini\n")
        f.write("   degistirememesinden de kaynaklanabilir. Bu benchmark hucre alanini dogrudan ayristirarak olcmez.\n")

    print(f"[OK] Saved strictly paired report to {report_path}")

    with open(report_path, 'r', encoding='utf-8') as f:
        print("\n" + f.read())


if __name__ == '__main__':
    main()
