"""
Phase 2 & Phase 3 Unified Paired Evaluation Pipeline (LDM v12 vs LDM v16 Stage 2 MSC)
=====================================================================================
Executes the strictly matched, zero-leakage evaluation across all 5 seeds (42, 1024, 2026, 7777, 9999)
plus the Phase 3 Trajectory Control Ablation on Seed 2026.

Strict Protocol Enforcement:
  1. Identical Test Set: data/processed_v4/test (Young N=327, Senescent N=318), 100% blinded during training.
  2. Identical Sampling Protocol:
     - Batch Size = 8, sorted() filenames, 100 DDIM steps
     - Full FP32 deterministic inference (torch.backends.cudnn.deterministic = True)
     - Identical per-batch CUDA RNG consumption: 1 VAE encode sample + 1 torch.randn_like(latents) per batch
  3. Matched Canonical Endpoint Parameters:
     - Rejuvenation: Strength = 0.65, CFG = 3.5
     - Aging:        Strength = 0.75, CFG = 4.0
  4. Provenance Manifests:
     - Every output folder writes manifest.json with checkpoint SHA256, epoch, best_val_loss, seed, precision, and parameters.
  5. Complete Multi-Metric 5-Seed Reporting:
     - Both FID_vs_Young and FID_vs_Senescent AND ResNet-18 Accuracy & Confidence (Mean ± Std) for both V12 and V16.
"""

import os
import sys
import gc
import json
import time
import shutil
import hashlib
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from pytorch_fid import fid_score

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    except Exception:
        pass

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
from models.classifier.classifier import Classifier
from src.inference.ldm.eval_v16_trajectory import run_trajectory

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def prevent_windows_sleep():
    """Prevents Windows from entering sleep or standby during long inference runs."""
    if sys.platform == 'win32':
        try:
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000040
            res = ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
            if res != 0:
                print("[*] Windows sleep/standby prevention successfully activated.", flush=True)
            else:
                print("[!] Notice: SetThreadExecutionState returned 0.", flush=True)
        except Exception as ex:
            print(f"[!] Warning: Could not activate Windows sleep prevention: {ex}", flush=True)

SEEDS = [42, 1024, 2026, 7777, 9999]
BATCH_SIZE = 8
DDIM_STEPS = 100
AGING_STRENGTH = 0.75
AGING_CFG = 4.0
REJUV_STRENGTH = 0.65
REJUV_CFG = 3.5

TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
V12_CKPT_PATH = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
V16_CKPT_PATH = os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v16_stage2_msc.pt')
CLASSIFIER_CKPT = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_v12_model():
    print(f"[*] Loading LDM v12 Checkpoint: {V12_CKPT_PATH}", flush=True)
    model = CellLDM(num_classes=2).to(DEVICE)
    ckpt = torch.load(V12_CKPT_PATH, map_location='cpu')
    epoch = ckpt.get('epoch', 'last')
    for key in ['unet_state_dict', 'ema_unet_state_dict']:
        if key in ckpt and 'class_embedding.weight' in ckpt[key]:
            w = ckpt[key]['class_embedding.weight']
            if w.shape[0] == 2:
                pad = torch.zeros(1, w.shape[1], device=w.device)
                ckpt[key]['class_embedding.weight'] = torch.cat([w, pad], dim=0)
    model.unet.load_state_dict(ckpt['unet_state_dict'])
    model.init_ema()
    if 'ema_unet_state_dict' in ckpt:
        model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
    model.eval()
    del ckpt
    gc.collect()
    return model, epoch


def generate_v12_all_seeds(force_regen=False):
    print("\n" + "=" * 80)
    print(" PHASE 2A: LDM v12 5-SEED PAIRED REGENERATION (FP32, BS=8, 100 DDIM STEPS) ")
    print("=" * 80, flush=True)

    v12_sha256 = get_sha256(V12_CKPT_PATH)
    transform_gen = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.lower().endswith(('.jpg', '.png'))])
    senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.lower().endswith(('.jpg', '.png'))])

    young_tensors = [transform_gen(Image.open(os.path.join(TEST_YOUNG, f)).convert('RGB')) for f in young_files]
    senes_tensors = [transform_gen(Image.open(os.path.join(TEST_SENES, f)).convert('RGB')) for f in senes_files]

    model, v12_epoch = load_v12_model()

    for seed in SEEDS:
        seed_dir = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', f'seed_{seed}')
        aging_dir = os.path.join(seed_dir, 'aging')
        rejuv_dir = os.path.join(seed_dir, 'rejuv')
        manifest_path = os.path.join(seed_dir, 'manifest.json')

        # Check if already freshly generated under paired FP32 BS=8 protocol
        if not force_regen and os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    mdata = json.load(mf)
                if (mdata.get('ckpt_sha256') == v12_sha256 and
                        mdata.get('batch_size') == BATCH_SIZE and
                        mdata.get('precision') == 'fp32' and
                        len(os.listdir(aging_dir)) == len(young_files) and
                        len(os.listdir(rejuv_dir)) == len(senes_files)):
                    print(f"[*] V12 Seed {seed} already generated with verified manifest. Skipping generation.", flush=True)
                    continue
            except Exception:
                pass

        # Clean old directories to avoid stale images
        if os.path.exists(aging_dir):
            shutil.rmtree(aging_dir)
        if os.path.exists(rejuv_dir):
            shutil.rmtree(rejuv_dir)
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(rejuv_dir, exist_ok=True)

        t0 = time.time()
        print(f"\n[*] Generating V12 Seed {seed} | Aging (N={len(young_files)}) & Rejuvenation (N={len(senes_files)})...", flush=True)

        # 1. Aging (Young -> Senescent, target=1)
        set_seed(seed)
        for i in range(0, len(young_files), BATCH_SIZE):
            b_files = young_files[i:i + BATCH_SIZE]
            b_tensor = torch.stack(young_tensors[i:i + BATCH_SIZE]).to(DEVICE)
            with torch.no_grad():
                fakes = model.translate(
                    b_tensor,
                    target_labels=torch.tensor([1] * len(b_files), device=DEVICE),
                    strength=AGING_STRENGTH,
                    num_steps=DDIM_STEPS,
                    guidance_scale=AGING_CFG,
                    use_ema=True
                )
            for f, fake in zip(b_files, fakes):
                save_image(fake.float().cpu().clamp(0, 1), os.path.join(aging_dir, f))

        # 2. Rejuvenation (Senescent -> Young, target=0)
        set_seed(seed)
        for i in range(0, len(senes_files), BATCH_SIZE):
            b_files = senes_files[i:i + BATCH_SIZE]
            b_tensor = torch.stack(senes_tensors[i:i + BATCH_SIZE]).to(DEVICE)
            with torch.no_grad():
                fakes = model.translate(
                    b_tensor,
                    target_labels=torch.tensor([0] * len(b_files), device=DEVICE),
                    strength=REJUV_STRENGTH,
                    num_steps=DDIM_STEPS,
                    guidance_scale=REJUV_CFG,
                    use_ema=True
                )
            for f, fake in zip(b_files, fakes):
                save_image(fake.float().cpu().clamp(0, 1), os.path.join(rejuv_dir, f))

        manifest = {
            'model': 'LDM_v12_LPIPS',
            'ckpt_path': os.path.abspath(V12_CKPT_PATH),
            'ckpt_filename': os.path.basename(V12_CKPT_PATH),
            'ckpt_sha256': v12_sha256,
            'ckpt_epoch': v12_epoch,
            'seed': seed,
            'batch_size': BATCH_SIZE,
            'ddim_steps': DDIM_STEPS,
            'precision': 'fp32',
            'deterministic': True,
            'aging_params': {'strength': AGING_STRENGTH, 'cfg': AGING_CFG, 'n_total': len(young_files)},
            'rejuv_params': {'strength': REJUV_STRENGTH, 'cfg': REJUV_CFG, 'n_total': len(senes_files)},
        }
        with open(manifest_path, 'w', encoding='utf-8') as mf:
            json.dump(manifest, mf, indent=2)

        print(f"[OK] V12 Seed {seed} completed in {time.time() - t0:.1f}s (manifest saved).", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def make_v16_args(seed, out_dir, progressive=True, fixed_passage_control=False):
    return argparse.Namespace(
        direction='both',
        ckpt=V16_CKPT_PATH,
        out_dir=out_dir,
        all=True,
        n_samples=None,
        batch_size=BATCH_SIZE,
        seed=seed,
        rejuv_strength=REJUV_STRENGTH,
        rejuv_cfg=REJUV_CFG,
        aging_strength=AGING_STRENGTH,
        aging_cfg=AGING_CFG,
        rejuv_start_strength=0.35,
        aging_start_strength=0.35,
        rejuv_start_cfg=2.0,
        aging_start_cfg=2.0,
        progressive=progressive,
        fixed_passage_control=fixed_passage_control,
        precision='fp32',
        calibrate_input=True,
        traj_mode='proportional',
        ddim_steps=DDIM_STEPS,
        match_v12_cohort=False,
        n_steps=5,
        delta_step=1.0,
        min_passage=3.5,
        max_passage=14.5,
        skip_generation=False
    )


def generate_v16_all_seeds():
    print("\n" + "=" * 80)
    print(" PHASE 2B: LDM v16 STAGE 2 (EPOCH 150 CLEAN) 5-SEED PAIRED EVALUATION ")
    print("=" * 80, flush=True)

    for seed in SEEDS:
        out_dir = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', f'epoch_150_clean_seed{seed}')
        args = make_v16_args(seed=seed, out_dir=out_dir, progressive=True, fixed_passage_control=False)
        for d in ['rejuvenation', 'aging']:
            dir_name = 'rejuvenation_trajectory' if d == 'rejuvenation' else 'aging_trajectory'
            pref = 'rejuvenation' if d == 'rejuvenation' else 'aging'
            traj_dir = os.path.join(out_dir, dir_name)
            summary_path = os.path.join(traj_dir, f'{pref}_trajectory_summary.csv')
            manifest_path = os.path.join(traj_dir, 'manifest.json')

            if os.path.exists(summary_path) and os.path.exists(manifest_path):
                print(f"[*] V16 Seed {seed} {d} already fully completed (summary CSV verified). Skipping.", flush=True)
                continue

            # If partial directory exists without complete summary, clean it up to ensure clean RNG determinism
            if os.path.exists(traj_dir):
                print(f"[*] Cleaning incomplete trajectory directory for V16 Seed {seed} {d}...", flush=True)
                shutil.rmtree(traj_dir)

            run_trajectory(d, args)


def evaluate_5seed_master_table():
    print("\n" + "=" * 80)
    print(" PHASE 2C: UNIFIED 5-SEED MASTER BENCHMARK (V12 vs V16 STAGE 2 CLEAN) ")
    print("=" * 80, flush=True)

    # Load ResNet-18 classifier
    cls_resnet = Classifier(output_size=2).to(DEVICE)
    cls_resnet.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE))
    cls_resnet.eval()
    imnet_mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    imnet_std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    @torch.no_grad()
    def eval_resnet_folder(folder, target_cls):
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        tensors = torch.stack([TF.to_tensor(Image.open(f).convert('RGB')) for f in files])
        correct = 0
        confs = []
        for i in range(0, len(tensors), 32):
            b = tensors[i:i + 32].to(DEVICE)
            g = (b - imnet_mean) / imnet_std
            probs = F.softmax(cls_resnet(g), dim=1)
            preds = torch.argmax(probs, dim=1)
            correct += (preds == target_cls).sum().item()
            confs.extend(probs[:, target_cls].cpu().numpy().tolist())
        return (correct / len(files)) * 100.0, float(np.mean(confs)) * 100.0

    # Load InceptionV3 once for exact Dual-Class FID
    block_idx = fid_score.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc_model = fid_score.InceptionV3([block_idx]).to(DEVICE).eval()

    young_ref = sorted([os.path.join(TEST_YOUNG, f) for f in os.listdir(TEST_YOUNG) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    senes_ref = sorted([os.path.join(TEST_SENES, f) for f in os.listdir(TEST_SENES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mu_y, sig_y = fid_score.calculate_activation_statistics(young_ref, inc_model, batch_size=32, dims=2048, device=DEVICE, num_workers=0)
    mu_s, sig_s = fid_score.calculate_activation_statistics(senes_ref, inc_model, batch_size=32, dims=2048, device=DEVICE, num_workers=0)

    def eval_fids_folder(folder):
        files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        mu, sig = fid_score.calculate_activation_statistics(files, inc_model, batch_size=32, dims=2048, device=DEVICE, num_workers=0)
        fid_y = float(fid_score.calculate_frechet_distance(mu, sig, mu_y, sig_y))
        fid_s = float(fid_score.calculate_frechet_distance(mu, sig, mu_s, sig_s))
        return fid_y, fid_s

    rows = []
    for seed in SEEDS:
        # 1. V12
        v12_base = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', f'seed_{seed}')
        v12_rejuv_dir = os.path.join(v12_base, 'rejuv')
        v12_aging_dir = os.path.join(v12_base, 'aging')

        v12_r_fidy, v12_r_fids = eval_fids_folder(v12_rejuv_dir)
        v12_r_acc, v12_r_conf = eval_resnet_folder(v12_rejuv_dir, target_cls=1)

        v12_a_fidy, v12_a_fids = eval_fids_folder(v12_aging_dir)
        v12_a_acc, v12_a_conf = eval_resnet_folder(v12_aging_dir, target_cls=0)

        rows.append({
            'Model': 'LDM_v12',
            'Seed': seed,
            'Rejuv_FID_vs_Young': round(v12_r_fidy, 2),
            'Rejuv_FID_vs_Senes': round(v12_r_fids, 2),
            'Rejuv_ResNet_Acc_pct': round(v12_r_acc, 2),
            'Rejuv_ResNet_Conf_pct': round(v12_r_conf, 2),
            'Aging_FID_vs_Senes': round(v12_a_fids, 2),
            'Aging_FID_vs_Young': round(v12_a_fidy, 2),
            'Aging_ResNet_Acc_pct': round(v12_a_acc, 2),
            'Aging_ResNet_Conf_pct': round(v12_a_conf, 2),
        })

        # 2. V16 Step 5 (Final Endpoint)
        v16_base = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', f'epoch_150_clean_seed{seed}')
        v16_rejuv_s5 = os.path.join(v16_base, 'rejuvenation_trajectory', 'gen_step5')
        v16_aging_s5 = os.path.join(v16_base, 'aging_trajectory', 'gen_step5')

        v16_r_fidy, v16_r_fids = eval_fids_folder(v16_rejuv_s5)
        v16_r_acc, v16_r_conf = eval_resnet_folder(v16_rejuv_s5, target_cls=1)

        v16_a_fidy, v16_a_fids = eval_fids_folder(v16_aging_s5)
        v16_a_acc, v16_a_conf = eval_resnet_folder(v16_aging_s5, target_cls=0)

        rows.append({
            'Model': 'LDM_v16_Stage2_Ep150',
            'Seed': seed,
            'Rejuv_FID_vs_Young': round(v16_r_fidy, 2),
            'Rejuv_FID_vs_Senes': round(v16_r_fids, 2),
            'Rejuv_ResNet_Acc_pct': round(v16_r_acc, 2),
            'Rejuv_ResNet_Conf_pct': round(v16_r_conf, 2),
            'Aging_FID_vs_Senes': round(v16_a_fids, 2),
            'Aging_FID_vs_Young': round(v16_a_fidy, 2),
            'Aging_ResNet_Acc_pct': round(v16_a_acc, 2),
            'Aging_ResNet_Conf_pct': round(v16_a_conf, 2),
        })
        print(f"  [Seed {seed}] V12 Rejuv FID={v12_r_fidy:.2f} (Acc={v12_r_acc:.1f}%) | V16 Rejuv FID={v16_r_fidy:.2f} (Acc={v16_r_acc:.1f}%) || V12 Aging FID={v12_a_fids:.2f} (Acc={v12_a_acc:.1f}%) | V16 Aging FID={v16_a_fids:.2f} (Acc={v16_a_acc:.1f}%)", flush=True)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', '5seed_benchmark_v12_vs_v16_clean.csv')
    df.to_csv(out_csv, index=False)

    # Compute summary stats (Mean ± Std)
    summary_rows = []
    metric_cols = [c for c in df.columns if c not in ('Model', 'Seed')]
    for model_name in ['LDM_v12', 'LDM_v16_Stage2_Ep150']:
        sub = df[df['Model'] == model_name]
        s_row = {'Model': model_name}
        for mc in metric_cols:
            vals = sub[mc].values
            s_row[f'{mc}_Mean'] = round(float(np.mean(vals)), 2)
            s_row[f'{mc}_Std'] = round(float(np.std(vals, ddof=1)), 2)
        summary_rows.append(s_row)

    df_sum = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', '5seed_summary_mean_std_v12_vs_v16.csv')
    df_sum.to_csv(sum_csv, index=False)

    print("\n" + "=" * 80)
    print(" 5-SEED SUMMARY (MEAN ± SAMPLE STD ACROSS N=5 SEEDS) ")
    print("=" * 80)
    for s_row in summary_rows:
        print(f"\nModel: {s_row['Model']}")
        print(f"  Rejuvenation -> FID_vs_Young: {s_row['Rejuv_FID_vs_Young_Mean']:.2f} ± {s_row['Rejuv_FID_vs_Young_Std']:.2f} | FID_vs_Senes: {s_row['Rejuv_FID_vs_Senes_Mean']:.2f} ± {s_row['Rejuv_FID_vs_Senes_Std']:.2f} | ResNet Acc: {s_row['Rejuv_ResNet_Acc_pct_Mean']:.2f}% ± {s_row['Rejuv_ResNet_Acc_pct_Std']:.2f}% | Conf: {s_row['Rejuv_ResNet_Conf_pct_Mean']:.2f}% ± {s_row['Rejuv_ResNet_Conf_pct_Std']:.2f}%")
        print(f"  Aging        -> FID_vs_Senes: {s_row['Aging_FID_vs_Senes_Mean']:.2f} ± {s_row['Aging_FID_vs_Senes_Std']:.2f} | FID_vs_Young: {s_row['Aging_FID_vs_Young_Mean']:.2f} ± {s_row['Aging_FID_vs_Young_Std']:.2f} | ResNet Acc: {s_row['Aging_ResNet_Acc_pct_Mean']:.2f}% ± {s_row['Aging_ResNet_Acc_pct_Std']:.2f}% | Conf: {s_row['Aging_ResNet_Conf_pct_Mean']:.2f}% ± {s_row['Aging_ResNet_Conf_pct_Std']:.2f}%")

    del cls_resnet, inc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def run_phase3_ablations():
    print("\n" + "=" * 80)
    print(" PHASE 3: TRAJECTORY CONTROL ABLATIONS (SEED 2026) ")
    print("=" * 80, flush=True)

    # Control 1: Fixed Strength/CFG (--no_progressive), varying target passage P (20% -> 100%)
    out_ctrl1 = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', 'ablation_seed2026_fixed_strength_cfg')
    args_ctrl1 = make_v16_args(seed=2026, out_dir=out_ctrl1, progressive=False, fixed_passage_control=False)
    for d in ['rejuvenation', 'aging']:
        dir_name = 'rejuvenation_trajectory' if d == 'rejuvenation' else 'aging_trajectory'
        pref = 'rejuvenation' if d == 'rejuvenation' else 'aging'
        traj_dir = os.path.join(out_ctrl1, dir_name)
        if os.path.exists(os.path.join(traj_dir, f'{pref}_trajectory_summary.csv')):
            print(f"[*] Ablation Control 1 (Fixed Strength/CFG) {d} already completed. Skipping.", flush=True)
            continue
        if os.path.exists(traj_dir):
            shutil.rmtree(traj_dir)
        run_trajectory(d, args_ctrl1)

    # Control 2: Fixed Target Passage (--fixed_passage_control), progressive Strength/CFG (0.35/2.0 -> endpoint)
    out_ctrl2 = os.path.join(root_dir, 'results', 'v16_stage2_evaluation', 'ablation_seed2026_fixed_passage')
    args_ctrl2 = make_v16_args(seed=2026, out_dir=out_ctrl2, progressive=True, fixed_passage_control=True)
    for d in ['rejuvenation', 'aging']:
        dir_name = 'rejuvenation_trajectory' if d == 'rejuvenation' else 'aging_trajectory'
        pref = 'rejuvenation' if d == 'rejuvenation' else 'aging'
        traj_dir = os.path.join(out_ctrl2, dir_name)
        if os.path.exists(os.path.join(traj_dir, f'{pref}_trajectory_summary.csv')):
            print(f"[*] Ablation Control 2 (Fixed Passage) {d} already completed. Skipping.", flush=True)
            continue
        if os.path.exists(traj_dir):
            shutil.rmtree(traj_dir)
        run_trajectory(d, args_ctrl2)


if __name__ == '__main__':
    prevent_windows_sleep()
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'v12', 'v16', 'metrics', 'ablation'], default='all')
    parser.add_argument('--force_v12', action='store_true', default=False)
    cli_args = parser.parse_args()

    if cli_args.phase in ('all', 'v12'):
        generate_v12_all_seeds(force_regen=cli_args.force_v12)
    if cli_args.phase in ('all', 'v16'):
        generate_v16_all_seeds()
    if cli_args.phase in ('all', 'metrics'):
        evaluate_5seed_master_table()
    if cli_args.phase in ('all', 'ablation'):
        run_phase3_ablations()

