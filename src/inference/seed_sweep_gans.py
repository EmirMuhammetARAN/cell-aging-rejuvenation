"""
5-Seed Sensitivity & Determinism Benchmark for CUT GAN (Epoch 200) and CycleGAN (Epoch 160)
=============================================================================================
Evaluates generative determinism, phenotypic transition accuracy, and distribution realism
across 5 independent random seeds: [42, 1024, 2026, 7777, 9999] on the held-out test cohort (N=645):
  - Aging: 327 young cells -> senescent (target: class 0)
  - Rejuvenation: 318 senescent cells -> young (target: class 1)

Models:
  1. CUT GAN Epoch 200: checkpoints/cut/cut_epoch_200.pth
  2. CycleGAN Epoch 160: checkpoints/cyclegan/cyclegan_v2_epoch_160.pth
  3. LDM v12 Latent-L1 FT (Total Ep 300 / Last): precomputed in results/generated/seed_sweep/

Metrics:
  - ResNet-18 Classifier Transition Accuracy (%) & Target Confidence (%)
  - Fréchet Inception Distance (FID) vs. ground-truth test reference cohorts
  - Cross-seed pixel discrepancy (L1 & Linf max delta)
  - Cross-seed Pairwise SSIM
  - Full population phenotypic consensus rate (% unanimous 5/5, % supermajority >= 4/5)
"""

import os
import sys
import gc
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid import fid_score
import skimage.metrics

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.cut.generator import CUTGenerator
from models.cyclegan.generator_resnet import GeneratorResNet
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.cyclegan_gan_model import CycleGANModel
from models.classifier.classifier import Classifier

# Hardware setup
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seeds
SEEDS = [42, 1024, 2026, 7777, 9999]

# Paths
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CLASSIFIER_CKPT = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')

CUT_CKPT = os.path.join(root_dir, 'checkpoints', 'cut', 'cut_epoch_200.pth')
CYCLEGAN_CKPT = os.path.join(root_dir, 'checkpoints', 'cyclegan', 'cyclegan_v2_epoch_160.pth')

CUT_OUTPUT_BASE = os.path.join(root_dir, 'results', 'generated', 'cut_seed_sweep')
CYCLEGAN_OUTPUT_BASE = os.path.join(root_dir, 'results', 'generated', 'cyclegan_seed_sweep')
LDM_SEED_BASE = os.path.join(root_dir, 'results', 'generated', 'seed_sweep')

RESULTS_JSON = os.path.join(root_dir, 'results', 'generated', 'gan_seed_sweep_results.json')
THREE_WAY_JSON = os.path.join(root_dir, 'results', 'generated', 'three_way_seed_invariance_comparison.json')

# Transforms
transform_gen = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_folder_classifier(folder_path, classifier, target_label):
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])
    correct = 0
    confs = []
    preds = {}
    with torch.no_grad():
        for fname in files:
            img_p = os.path.join(folder_path, fname)
            img = cls_transform(Image.open(img_p).convert('RGB')).unsqueeze(0).to(DEVICE)
            probs = torch.softmax(classifier(img), dim=1)
            pred = probs.argmax(dim=1).item()
            conf = probs[0, target_label].item() * 100.0
            confs.append(conf)
            preds[fname] = {'pred': pred, 'conf': conf}
            if pred == target_label:
                correct += 1
    acc = (correct / len(files)) * 100.0 if files else 0.0
    mean_conf = float(np.mean(confs)) if confs else 0.0
    std_conf = float(np.std(confs)) if confs else 0.0
    return acc, mean_conf, std_conf, preds


def run_cut_seed_sweep(classifier, young_files, senes_files):
    print("\n" + "=" * 80)
    print("STARTING CUT GAN (EPOCH 200) 5-SEED EVALUATION")
    print("=" * 80)
    os.makedirs(CUT_OUTPUT_BASE, exist_ok=True)

    # Load CUT Generator once
    aging_G = CUTGenerator(num_residual_blocks=9)
    reju_G = CUTGenerator(num_residual_blocks=9)
    ckpt = torch.load(CUT_CKPT, map_location=DEVICE)
    aging_G.load_state_dict(ckpt['aging_G'])
    reju_G.load_state_dict(ckpt['reju_G'])
    aging_G.to(DEVICE, memory_format=torch.channels_last).eval()
    reju_G.to(DEVICE, memory_format=torch.channels_last).eval()

    seed_results = []
    all_seed_preds_aging = {}
    all_seed_preds_reju = {}

    for seed in SEEDS:
        print(f"\n---> [CUT GAN] Processing Seed {seed}...")
        set_seed(seed)
        seed_dir = os.path.join(CUT_OUTPUT_BASE, f"seed_{seed}")
        aging_dir = os.path.join(seed_dir, 'aging')
        reju_dir = os.path.join(seed_dir, 'rejuv')
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(reju_dir, exist_ok=True)

        # 1. Generate Aging (Young -> Senescent)
        for fname in tqdm(young_files, desc=f"CUT Ep200 Aging (Seed {seed})", leave=False):
            out_p = os.path.join(aging_dir, fname)
            if not os.path.exists(out_p):
                img = transform_gen(Image.open(os.path.join(TEST_YOUNG, fname)).convert('RGB')).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        fake = aging_G(img)
                fake = (fake.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
                transforms.ToPILImage()(fake).save(out_p)

        # 2. Generate Rejuvenation (Senescent -> Young)
        for fname in tqdm(senes_files, desc=f"CUT Ep200 Reju (Seed {seed})", leave=False):
            out_p = os.path.join(reju_dir, fname)
            if not os.path.exists(out_p):
                img = transform_gen(Image.open(os.path.join(TEST_SENES, fname)).convert('RGB')).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        fake = reju_G(img)
                fake = (fake.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
                transforms.ToPILImage()(fake).save(out_p)

        # 3. Classifier Evaluation
        acc_a, conf_a, std_conf_a, preds_a = evaluate_folder_classifier(aging_dir, classifier, target_label=0)
        acc_r, conf_r, std_conf_r, preds_r = evaluate_folder_classifier(reju_dir, classifier, target_label=1)
        all_seed_preds_aging[seed] = preds_a
        all_seed_preds_reju[seed] = preds_r

        overall_acc = ((acc_a * len(young_files)) + (acc_r * len(senes_files))) / (len(young_files) + len(senes_files))
        overall_conf = ((conf_a * len(young_files)) + (conf_r * len(senes_files))) / (len(young_files) + len(senes_files))

        # 4. FID Evaluation
        fid_a = fid_score.calculate_fid_given_paths([TEST_SENES, aging_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)
        fid_r = fid_score.calculate_fid_given_paths([TEST_YOUNG, reju_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)

        print(f"     Seed {seed} -> Aging ACC: {acc_a:.2f}%, Reju ACC: {acc_r:.2f}%, Overall ACC: {overall_acc:.2f}%, Aging FID: {fid_a:.2f}, Reju FID: {fid_r:.2f}")

        seed_results.append({
            'seed': seed,
            'aging_acc': acc_a,
            'aging_conf': conf_a,
            'aging_fid': fid_a,
            'reju_acc': acc_r,
            'reju_conf': conf_r,
            'reju_fid': fid_r,
            'overall_acc': overall_acc,
            'overall_conf': overall_conf
        })

    del aging_G, reju_G, ckpt
    torch.cuda.empty_cache()
    gc.collect()

    return seed_results, all_seed_preds_aging, all_seed_preds_reju


def run_cyclegan_seed_sweep(classifier, young_files, senes_files):
    print("\n" + "=" * 80)
    print("STARTING CYCLEGAN (EPOCH 160) 5-SEED EVALUATION")
    print("=" * 80)
    os.makedirs(CYCLEGAN_OUTPUT_BASE, exist_ok=True)

    # Load CycleGAN Model once
    model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE, use_lpips=False)
    checkpoint = torch.load(CYCLEGAN_CKPT, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE, memory_format=torch.channels_last).eval()

    seed_results = []
    all_seed_preds_aging = {}
    all_seed_preds_reju = {}

    for seed in SEEDS:
        print(f"\n---> [CycleGAN] Processing Seed {seed}...")
        set_seed(seed)
        seed_dir = os.path.join(CYCLEGAN_OUTPUT_BASE, f"seed_{seed}")
        aging_dir = os.path.join(seed_dir, 'aging')
        reju_dir = os.path.join(seed_dir, 'rejuv')
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(reju_dir, exist_ok=True)

        # 1. Generate Aging (Young -> Senescent)
        for fname in tqdm(young_files, desc=f"CycleGAN Ep160 Aging (Seed {seed})", leave=False):
            out_p = os.path.join(aging_dir, fname)
            if not os.path.exists(out_p):
                img = transform_gen(Image.open(os.path.join(TEST_YOUNG, fname)).convert('RGB')).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        fake = model.G_AB(img)
                fake = (fake.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
                transforms.ToPILImage()(fake).save(out_p)

        # 2. Generate Rejuvenation (Senescent -> Young)
        for fname in tqdm(senes_files, desc=f"CycleGAN Ep160 Reju (Seed {seed})", leave=False):
            out_p = os.path.join(reju_dir, fname)
            if not os.path.exists(out_p):
                img = transform_gen(Image.open(os.path.join(TEST_SENES, fname)).convert('RGB')).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        fake = model.G_BA(img)
                fake = (fake.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
                transforms.ToPILImage()(fake).save(out_p)

        # 3. Classifier Evaluation
        acc_a, conf_a, std_conf_a, preds_a = evaluate_folder_classifier(aging_dir, classifier, target_label=0)
        acc_r, conf_r, std_conf_r, preds_r = evaluate_folder_classifier(reju_dir, classifier, target_label=1)
        all_seed_preds_aging[seed] = preds_a
        all_seed_preds_reju[seed] = preds_r

        overall_acc = ((acc_a * len(young_files)) + (acc_r * len(senes_files))) / (len(young_files) + len(senes_files))
        overall_conf = ((conf_a * len(young_files)) + (conf_r * len(senes_files))) / (len(young_files) + len(senes_files))

        # 4. FID Evaluation
        fid_a = fid_score.calculate_fid_given_paths([TEST_SENES, aging_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)
        fid_r = fid_score.calculate_fid_given_paths([TEST_YOUNG, reju_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)

        print(f"     Seed {seed} -> Aging ACC: {acc_a:.2f}%, Reju ACC: {acc_r:.2f}%, Overall ACC: {overall_acc:.2f}%, Aging FID: {fid_a:.2f}, Reju FID: {fid_r:.2f}")

        seed_results.append({
            'seed': seed,
            'aging_acc': acc_a,
            'aging_conf': conf_a,
            'aging_fid': fid_a,
            'reju_acc': acc_r,
            'reju_conf': conf_r,
            'reju_fid': fid_r,
            'overall_acc': overall_acc,
            'overall_conf': overall_conf
        })

    del model, checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return seed_results, all_seed_preds_aging, all_seed_preds_reju


def evaluate_ldm_seed_sweep(classifier, young_files, senes_files):
    print("\n" + "=" * 80)
    print("EVALUATING EXISTING LDM v12 (EPOCH 300 / LAST) 5-SEED RESULTS")
    print("=" * 80)

    seed_results = []
    all_seed_preds_aging = {}
    all_seed_preds_reju = {}

    for seed in SEEDS:
        seed_dir = os.path.join(LDM_SEED_BASE, f"seed_{seed}")
        aging_dir = os.path.join(seed_dir, 'aging')
        reju_dir = os.path.join(seed_dir, 'rejuv')

        if not os.path.exists(aging_dir) or not os.path.exists(reju_dir):
            print(f"[!] Warning: LDM seed {seed} directory not found at {seed_dir}")
            continue

        acc_a, conf_a, std_conf_a, preds_a = evaluate_folder_classifier(aging_dir, classifier, target_label=0)
        acc_r, conf_r, std_conf_r, preds_r = evaluate_folder_classifier(reju_dir, classifier, target_label=1)
        all_seed_preds_aging[seed] = preds_a
        all_seed_preds_reju[seed] = preds_r

        overall_acc = ((acc_a * len(young_files)) + (acc_r * len(senes_files))) / (len(young_files) + len(senes_files))
        overall_conf = ((conf_a * len(young_files)) + (conf_r * len(senes_files))) / (len(young_files) + len(senes_files))

        fid_a = fid_score.calculate_fid_given_paths([TEST_SENES, aging_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)
        fid_r = fid_score.calculate_fid_given_paths([TEST_YOUNG, reju_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)

        print(f"     LDM Seed {seed} -> Aging ACC: {acc_a:.2f}%, Reju ACC: {acc_r:.2f}%, Overall ACC: {overall_acc:.2f}%, Aging FID: {fid_a:.2f}, Reju FID: {fid_r:.2f}")

        seed_results.append({
            'seed': seed,
            'aging_acc': acc_a,
            'aging_conf': conf_a,
            'aging_fid': fid_a,
            'reju_acc': acc_r,
            'reju_conf': conf_r,
            'reju_fid': fid_r,
            'overall_acc': overall_acc,
            'overall_conf': overall_conf
        })

    return seed_results, all_seed_preds_aging, all_seed_preds_reju


def compute_cross_seed_determinism(base_dir, file_list, subfolder):
    """Verifies whether generated images across seeds are pixel-identical or stochastic."""
    max_l1 = 0.0
    max_linf = 0.0
    ssim_list = []

    # Check pair: seed 42 vs seed 9999
    dir_42 = os.path.join(base_dir, "seed_42", subfolder)
    dir_9999 = os.path.join(base_dir, "seed_9999", subfolder)

    for fname in file_list:
        p1 = os.path.join(dir_42, fname)
        p2 = os.path.join(dir_9999, fname)
        if os.path.exists(p1) and os.path.exists(p2):
            im1 = np.array(Image.open(p1).convert('RGB')).astype(np.float32)
            im2 = np.array(Image.open(p2).convert('RGB')).astype(np.float32)
            diff = np.abs(im1 - im2)
            max_l1 = max(max_l1, float(np.mean(diff)))
            max_linf = max(max_linf, float(np.max(diff)))
            s = skimage.metrics.structural_similarity(im1, im2, channel_axis=2, data_range=255)
            ssim_list.append(s)

    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 1.0
    return max_l1, max_linf, mean_ssim


def compute_consensus(all_seed_preds, file_list, target_label):
    unanimous = 0
    supermaj = 0
    for fname in file_list:
        correct_count = sum(1 for s in SEEDS if all_seed_preds[s][fname]['pred'] == target_label)
        if correct_count == 5:
            unanimous += 1
        if correct_count >= 4:
            supermaj += 1
    total = len(file_list)
    return (unanimous / total) * 100.0, (supermaj / total) * 100.0


def aggregate_stats(results):
    return {
        'aging_acc_mean': float(np.mean([r['aging_acc'] for r in results])),
        'aging_acc_std': float(np.std([r['aging_acc'] for r in results])),
        'reju_acc_mean': float(np.mean([r['reju_acc'] for r in results])),
        'reju_acc_std': float(np.std([r['reju_acc'] for r in results])),
        'overall_acc_mean': float(np.mean([r['overall_acc'] for r in results])),
        'overall_acc_std': float(np.std([r['overall_acc'] for r in results])),
        'aging_fid_mean': float(np.mean([r['aging_fid'] for r in results])),
        'aging_fid_std': float(np.std([r['aging_fid'] for r in results])),
        'reju_fid_mean': float(np.mean([r['reju_fid'] for r in results])),
        'reju_fid_std': float(np.std([r['reju_fid'] for r in results])),
    }


def main():
    print("=" * 80)
    print("      5-SEED BENCHMARK SUITE: CUT GAN vs CycleGAN vs LDM      ")
    print("=" * 80)
    print(f"Seeds: {SEEDS}")
    print(f"Device: {DEVICE}")

    # Load Classifier v2
    classifier = Classifier(output_size=2).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE))
    classifier.eval()
    print("[OK] Loaded ResNet-18 Classifier v2.")

    young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.lower().endswith(('.jpg', '.png'))])
    senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.lower().endswith(('.jpg', '.png'))])
    print(f"Cohort size: Aging = {len(young_files)} cells, Rejuvenation = {len(senes_files)} cells (Total = {len(young_files)+len(senes_files)})")

    # 1. CUT GAN
    cut_results, cut_preds_a, cut_preds_r = run_cut_seed_sweep(classifier, young_files, senes_files)
    cut_stats = aggregate_stats(cut_results)
    cut_l1_a, cut_linf_a, cut_ssim_a = compute_cross_seed_determinism(CUT_OUTPUT_BASE, young_files, 'aging')
    cut_l1_r, cut_linf_r, cut_ssim_r = compute_cross_seed_determinism(CUT_OUTPUT_BASE, senes_files, 'rejuv')
    cut_unan_a, cut_sup_a = compute_consensus(cut_preds_a, young_files, 0)
    cut_unan_r, cut_sup_r = compute_consensus(cut_preds_r, senes_files, 1)

    # 2. CycleGAN
    cycle_results, cycle_preds_a, cycle_preds_r = run_cyclegan_seed_sweep(classifier, young_files, senes_files)
    cycle_stats = aggregate_stats(cycle_results)
    cycle_l1_a, cycle_linf_a, cycle_ssim_a = compute_cross_seed_determinism(CYCLEGAN_OUTPUT_BASE, young_files, 'aging')
    cycle_l1_r, cycle_linf_r, cycle_ssim_r = compute_cross_seed_determinism(CYCLEGAN_OUTPUT_BASE, senes_files, 'rejuv')
    cycle_unan_a, cycle_sup_a = compute_consensus(cycle_preds_a, young_files, 0)
    cycle_unan_r, cycle_sup_r = compute_consensus(cycle_preds_r, senes_files, 1)

    # 3. LDM
    ldm_results, ldm_preds_a, ldm_preds_r = evaluate_ldm_seed_sweep(classifier, young_files, senes_files)
    ldm_stats = aggregate_stats(ldm_results)
    ldm_l1_a, ldm_linf_a, ldm_ssim_a = compute_cross_seed_determinism(LDM_SEED_BASE, young_files, 'aging')
    ldm_l1_r, ldm_linf_r, ldm_ssim_r = compute_cross_seed_determinism(LDM_SEED_BASE, senes_files, 'rejuv')
    ldm_unan_a, ldm_sup_a = compute_consensus(ldm_preds_a, young_files, 0)
    ldm_unan_r, ldm_sup_r = compute_consensus(ldm_preds_r, senes_files, 1)

    # Compile Final Structured Output
    full_output = {
        'seeds': SEEDS,
        'cohort': {'aging_young_cells': len(young_files), 'rejuv_senescent_cells': len(senes_files)},
        'cut_gan_epoch_200': {
            'per_seed': cut_results,
            'summary': cut_stats,
            'determinism': {
                'aging_l1_diff': cut_l1_a, 'aging_linf_diff': cut_linf_a, 'aging_pairwise_ssim': cut_ssim_a,
                'reju_l1_diff': cut_l1_r, 'reju_linf_diff': cut_linf_r, 'reju_pairwise_ssim': cut_ssim_r,
                'aging_unanimous_5_of_5': cut_unan_a, 'aging_supermajority_4_of_5': cut_sup_a,
                'reju_unanimous_5_of_5': cut_unan_r, 'reju_supermajority_4_of_5': cut_sup_r
            }
        },
        'cyclegan_epoch_160': {
            'per_seed': cycle_results,
            'summary': cycle_stats,
            'determinism': {
                'aging_l1_diff': cycle_l1_a, 'aging_linf_diff': cycle_linf_a, 'aging_pairwise_ssim': cycle_ssim_a,
                'reju_l1_diff': cycle_l1_r, 'reju_linf_diff': cycle_linf_r, 'reju_pairwise_ssim': cycle_ssim_r,
                'aging_unanimous_5_of_5': cycle_unan_a, 'aging_supermajority_4_of_5': cycle_sup_a,
                'reju_unanimous_5_of_5': cycle_unan_r, 'reju_supermajority_4_of_5': cycle_sup_r
            }
        },
        'ldm_v12_epoch_300': {
            'per_seed': ldm_results,
            'summary': ldm_stats,
            'determinism': {
                'aging_l1_diff': ldm_l1_a, 'aging_linf_diff': ldm_linf_a, 'aging_pairwise_ssim': ldm_ssim_a,
                'reju_l1_diff': ldm_l1_r, 'reju_linf_diff': ldm_linf_r, 'reju_pairwise_ssim': ldm_ssim_r,
                'aging_unanimous_5_of_5': ldm_unan_a, 'aging_supermajority_4_of_5': ldm_sup_a,
                'reju_unanimous_5_of_5': ldm_unan_r, 'reju_supermajority_4_of_5': ldm_sup_r
            }
        }
    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(full_output, f, indent=2)
    with open(THREE_WAY_JSON, 'w') as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 95)
    print("                    MASTER 3-WAY 5-SEED COMPARISON TABLE                     ")
    print("=" * 95)
    print(f"{'Model':<24} | {'Overall ACC (%)':<18} | {'Aging FID':<16} | {'Rejuv FID':<16} | {'Pairwise SSIM':<14}")
    print("-" * 95)
    print(f"{'CUT GAN (Ep 200)':<24} | {cut_stats['overall_acc_mean']:.2f} ± {cut_stats['overall_acc_std']:.2f}%     | {cut_stats['aging_fid_mean']:.2f} ± {cut_stats['aging_fid_std']:.2f}     | {cut_stats['reju_fid_mean']:.2f} ± {cut_stats['reju_fid_std']:.2f}     | {cut_ssim_a:.4f} (det.)")
    print(f"{'CycleGAN (Ep 160)':<24} | {cycle_stats['overall_acc_mean']:.2f} ± {cycle_stats['overall_acc_std']:.2f}%     | {cycle_stats['aging_fid_mean']:.2f} ± {cycle_stats['aging_fid_std']:.2f}     | {cycle_stats['reju_fid_mean']:.2f} ± {cycle_stats['reju_fid_std']:.2f}     | {cycle_ssim_a:.4f} (det.)")
    print(f"{'LDM v12 (Ep 300)':<24} | {ldm_stats['overall_acc_mean']:.2f} ± {ldm_stats['overall_acc_std']:.2f}%     | {ldm_stats['aging_fid_mean']:.2f} ± {ldm_stats['aging_fid_std']:.2f}     | {ldm_stats['reju_fid_mean']:.2f} ± {ldm_stats['reju_fid_std']:.2f}     | {ldm_ssim_a:.4f} (stoch.)")
    print("=" * 95)
    print(f"\n[OK] Results saved to {RESULTS_JSON}")


if __name__ == '__main__':
    main()
