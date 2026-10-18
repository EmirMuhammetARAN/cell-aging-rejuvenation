# -*- coding: utf-8 -*-
"""
LDM v13 Full Test Set Evaluation - Matched v12 Hyperparameters (No Guidance)
=============================================================================
Directly tests LDM v13 without auxiliary classifier guidance using the exact
same inference hyperparameters (Strength and CFG) established for LDM v12:

- Aging (Young -> Senescent):
    * strength = 0.75
    * CFG = 4.0
    * num_steps = 50
    * target = Senescent endpoint (target_passage = 41.0 / score = 1.0)

- Rejuvenation (Senescent -> Young):
    * strength = 0.65
    * CFG = 3.5
    * num_steps = 50
    * target = Young endpoint (target_passage = 1.0 / score = 0.0)

Evaluates:
- Frechet Inception Distance (FID) vs real test ground truth
- Classifier v2 (ResNet-18) Transition Accuracy (%) & Confidence (%)
- Classifier v3 (ConvNeXt-Tiny) Biological Age Shifts (Passages)
- Grand Side-by-Side Comparison against LDM v12 baseline
"""

import os
import sys
import gc
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_v13 import CellLDM_v13
from models.classifier.classifier import Classifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Paths
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CHECKPOINT_LDM_V13 = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v13_continuous_age_last.pt')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm_v13_matched_v12')
AGING_OUT = os.path.join(OUTPUT_DIR, 'aging')
REJU_OUT = os.path.join(OUTPUT_DIR, 'rejuvenation')

os.makedirs(AGING_OUT, exist_ok=True)
os.makedirs(REJU_OUT, exist_ok=True)

# Transforms
transform_gen = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

v3_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class ConvNeXtMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        in_features = base.classifier[2].in_features
        self.norm = base.classifier[0]
        self.age_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.phase_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.norm(self.avgpool(self.features(x)))
        return self.age_head(x).squeeze(1), self.phase_head(x)


def run_generation(batch_size=4):
    print(f"\n{'='*75}")
    print(f"[1/4] Running LDM v13 Generation with Matched v12 Parameters (Unguided)")
    print(f"{'='*75}")
    print(f"Checkpoint: {CHECKPOINT_LDM_V13}")
    print(f"Aging Config:        strength = 0.75, CFG = 4.0, num_steps = 50, target_passage = 41.0 (score 1.0)")
    print(f"Rejuvenation Config: strength = 0.65, CFG = 3.5, num_steps = 50, target_passage = 1.0  (score 0.0)")

    model = CellLDM_v13()
    ckpt = torch.load(CHECKPOINT_LDM_V13, map_location='cpu')
    model.unet.load_state_dict(ckpt['unet_state_dict'])
    if 'ema_unet_state_dict' in ckpt:
        model.init_ema()
        model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
    model.to(DEVICE)
    model.eval()

    # 1. Aging: 327 young cells -> senescent
    young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.endswith(('.jpg', '.png'))])
    print(f"\n[1a] Generating Aging Translations: {len(young_files)} cells...")

    for i in tqdm(range(0, len(young_files), batch_size), desc="Aging (str=0.75, cfg=4.0)"):
        b_files = young_files[i:i+batch_size]
        imgs = [transform_gen(Image.open(os.path.join(TEST_YOUNG, f)).convert('RGB')) for f in b_files]
        b_tensor = torch.stack(imgs).to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                fakes = model.translate_continuous(
                    b_tensor,
                    target_passage=41.0, # score = 1.0 (Senescent anchor)
                    strength=0.75,
                    num_steps=50,
                    guidance_scale=4.0,
                    use_ema=True
                )

        for f, fake in zip(b_files, fakes):
            out_p = os.path.join(AGING_OUT, f)
            transforms.ToPILImage()(fake.float().cpu().clamp(0, 1)).save(out_p)

    # 2. Rejuvenation: 318 senescent cells -> young
    senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.endswith(('.jpg', '.png'))])
    print(f"\n[1b] Generating Rejuvenation Translations: {len(senes_files)} cells...")

    for i in tqdm(range(0, len(senes_files), batch_size), desc="Rejuvenation (str=0.65, cfg=3.5)"):
        b_files = senes_files[i:i+batch_size]
        imgs = [transform_gen(Image.open(os.path.join(TEST_SENES, f)).convert('RGB')) for f in b_files]
        b_tensor = torch.stack(imgs).to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                fakes = model.translate_continuous(
                    b_tensor,
                    target_passage=1.0, # score = 0.0 (Young anchor)
                    strength=0.65,
                    num_steps=50,
                    guidance_scale=3.5,
                    use_ema=True
                )

        for f, fake in zip(b_files, fakes):
            out_p = os.path.join(REJU_OUT, f)
            transforms.ToPILImage()(fake.float().cpu().clamp(0, 1)).save(out_p)

    del model, ckpt
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n[OK] Generation complete! Output directories:\n  Aging:        {AGING_OUT}\n  Rejuvenation: {REJU_OUT}")


def evaluate_classifier_v2():
    print(f"\n{'='*75}")
    print("[2/4] Evaluating Phenotypic Transition with Classifier v2 (ResNet-18)")
    print(f"{'='*75}")

    cls_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier = Classifier(output_size=2).to(DEVICE)
    classifier.load_state_dict(torch.load(cls_path, map_location=DEVICE))
    classifier.eval()

    # Aging: target is Senescent (Class 0)
    aging_files = sorted([f for f in os.listdir(AGING_OUT) if f.endswith(('.jpg', '.png'))])
    correct_a = 0
    confs_a = []
    for f in tqdm(aging_files, desc="Classifying Aging"):
        img = Image.open(os.path.join(AGING_OUT, f)).convert('RGB')
        t_img = cls_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier(t_img)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf_target = probs[0, 0].item()
            confs_a.append(conf_target)
            if pred == 0:
                correct_a += 1

    acc_a = (correct_a / len(aging_files)) * 100.0
    mean_conf_a = sum(confs_a) / len(confs_a) * 100.0

    # Rejuvenation: target is Young (Class 1)
    reju_files = sorted([f for f in os.listdir(REJU_OUT) if f.endswith(('.jpg', '.png'))])
    correct_r = 0
    confs_r = []
    for f in tqdm(reju_files, desc="Classifying Rejuvenation"):
        img = Image.open(os.path.join(REJU_OUT, f)).convert('RGB')
        t_img = cls_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = classifier(t_img)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(dim=1).item()
            conf_target = probs[0, 1].item()
            confs_r.append(conf_target)
            if pred == 1:
                correct_r += 1

    acc_r = (correct_r / len(reju_files)) * 100.0
    mean_conf_r = sum(confs_r) / len(confs_r) * 100.0

    total_cells = len(aging_files) + len(reju_files)
    total_correct = correct_a + correct_r
    overall_acc = (total_correct / total_cells) * 100.0
    all_confs = confs_a + confs_r
    overall_conf = sum(all_confs) / len(all_confs) * 100.0

    print(f"\n--- Classifier v2 (ResNet-18) Results ---")
    print(f"Aging Accuracy:        {correct_a}/{len(aging_files)} = {acc_a:.2f}% (Mean Target Conf: {mean_conf_a:.2f}%)")
    print(f"Rejuvenation Accuracy: {correct_r}/{len(reju_files)} = {acc_r:.2f}% (Mean Target Conf: {mean_conf_r:.2f}%)")
    print(f"Overall Transition ACC:{total_correct}/{total_cells} = {overall_acc:.2f}% (Mean Conf: {overall_conf:.2f}%)")

    del classifier
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'aging_acc': acc_a, 'aging_conf': mean_conf_a, 'correct_a': correct_a, 'total_a': len(aging_files),
        'reju_acc': acc_r, 'reju_conf': mean_conf_r, 'correct_r': correct_r, 'total_r': len(reju_files),
        'overall_acc': overall_acc, 'overall_conf': overall_conf, 'total_correct': total_correct, 'total_cells': total_cells
    }


def evaluate_fid():
    print(f"\n{'='*75}")
    print("[3/4] Evaluating Distribution Realism via FID (Fréchet Inception Distance)")
    print(f"{'='*75}")

    print("Computing Aging FID (Generated Aging vs Real Test Senescent)...")
    fid_a = fid_score.calculate_fid_given_paths(
        [TEST_SENES, AGING_OUT],
        batch_size=32,
        device=DEVICE,
        dims=2048,
        num_workers=0
    )
    print(f"-> LDM v13 Aging FID: {fid_a:.2f}")

    print("Computing Rejuvenation FID (Generated Rejuvenation vs Real Test Young)...")
    fid_r = fid_score.calculate_fid_given_paths(
        [TEST_YOUNG, REJU_OUT],
        batch_size=32,
        device=DEVICE,
        dims=2048,
        num_workers=0
    )
    print(f"-> LDM v13 Rejuvenation FID: {fid_r:.2f}")

    return {'aging_fid': fid_a, 'reju_fid': fid_r}


def evaluate_classifier_v3():
    v3_ckpt = os.path.join(root_dir, 'deneme', 'checkpoints', 'classifier_v3_convnext_best.pth')
    if not os.path.exists(v3_ckpt):
        print("\nClassifier v3 checkpoint not found, skipping.")
        return None

    print(f"\n{'='*75}")
    print("[4/4] Biological Evaluation with Classifier v3 (ConvNeXt-Tiny Biological Clock)")
    print(f"{'='*75}")

    model_v3 = ConvNeXtMultiTask().to(DEVICE)
    ckpt = torch.load(v3_ckpt, map_location=DEVICE)
    model_v3.load_state_dict(ckpt['model_state_dict'])
    model_v3.eval()

    def get_ages(folder):
        scores = []
        files = sorted([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))])
        for f in files:
            img = Image.open(os.path.join(folder, f)).convert('RGB')
            t = v3_transform(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_c, _ = model_v3(t)
                scores.append(pred_c.item())
        passages = [s * 40.0 + 1.0 for s in scores]
        return scores, passages

    _, real_y_p = get_ages(TEST_YOUNG)
    _, real_s_p = get_ages(TEST_SENES)
    _, v13_aging_p = get_ages(AGING_OUT)
    _, v13_reju_p = get_ages(REJU_OUT)

    mean_real_y = sum(real_y_p) / len(real_y_p)
    mean_real_s = sum(real_s_p) / len(real_s_p)
    mean_aging = sum(v13_aging_p) / len(v13_aging_p)
    mean_reju = sum(v13_reju_p) / len(v13_reju_p)

    delta_aging = mean_aging - mean_real_y
    delta_reju = mean_reju - mean_real_s

    print(f"Real Young Control Age:     P{mean_real_y:.2f}")
    print(f"LDM v13 Synthesized Aged:   P{mean_aging:.2f} (Shift: {delta_aging:+.2f} passages)")
    print(f"Real Senescent Control Age: P{mean_real_s:.2f}")
    print(f"LDM v13 Synthesized Reju:   P{mean_reju:.2f} (Shift: {delta_reju:+.2f} passages)")

    del model_v3
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'mean_real_young_passage': mean_real_y,
        'mean_v13_aging_passage': mean_aging,
        'aging_passage_shift': delta_aging,
        'mean_real_senes_passage': mean_real_s,
        'mean_v13_reju_passage': mean_reju,
        'reju_passage_shift': delta_reju,
    }


def main():
    print("=" * 80)
    print("      LDM v13 BENCHMARK: MATCHED v12 HYPERPARAMETERS (UN-GUIDED)       ")
    print("=" * 80)

    run_generation()
    cls2_res = evaluate_classifier_v2()
    fid_res = evaluate_fid()
    cls3_res = evaluate_classifier_v3()

    # Save metrics JSON
    summary = {
        'model': 'LDM v13 (Matched v12 Parameters, No Guidance)',
        'checkpoint': CHECKPOINT_LDM_V13,
        'hyperparameters': {
            'aging': {'strength': 0.75, 'cfg': 4.0, 'num_steps': 50, 'target_passage': 41.0},
            'rejuvenation': {'strength': 0.65, 'cfg': 3.5, 'num_steps': 50, 'target_passage': 1.0}
        },
        'metrics': {
            'aging_fid': fid_res['aging_fid'],
            'rejuvenation_fid': fid_res['reju_fid'],
            'aging_acc_v2': cls2_res['aging_acc'],
            'aging_conf_v2': cls2_res['aging_conf'],
            'reju_acc_v2': cls2_res['reju_acc'],
            'reju_conf_v2': cls2_res['reju_conf'],
            'overall_acc_v2': cls2_res['overall_acc'],
            'overall_conf_v2': cls2_res['overall_conf'],
            'biological_clock_v3': cls3_res
        }
    }

    summary_file = os.path.join(OUTPUT_DIR, 'ldm_v13_matched_metrics.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)
    print(f"\n[OK] Metrics saved to: {summary_file}")

    # Baseline v12 Reference values
    # From master benchmark: Aging FID 47.30, Rejuv FID 53.49, Aging ACC 89.60%, Rejuv ACC 88.05%, Overall 88.84%
    v12_aging_fid = 47.30
    v12_reju_fid = 53.49
    v12_aging_acc = 89.60
    v12_reju_acc = 88.05
    v12_overall_acc = 88.84

    delta_aging_fid = fid_res['aging_fid'] - v12_aging_fid
    delta_reju_fid = fid_res['reju_fid'] - v12_reju_fid
    delta_aging_acc = cls2_res['aging_acc'] - v12_aging_acc
    delta_reju_acc = cls2_res['reju_acc'] - v12_reju_acc
    delta_overall_acc = cls2_res['overall_acc'] - v12_overall_acc

    print(f"\n{'='*95}")
    print("                    DIRECT COMPARISON: LDM v12 vs. LDM v13 (MATCHED PARAMS)           ")
    print(f"{'='*95}")
    print(f"{'Metric':<25} | {'LDM v12 (Baseline)':<20} | {'LDM v13 (Matched)':<20} | {'Delta (v13 - v12)':<20}")
    print(f"{'-'*95}")
    print(f"{'Aging FID (lower best)':<25} | {v12_aging_fid:<20.2f} | {fid_res['aging_fid']:<20.2f} | {delta_aging_fid:<+20.2f}")
    print(f"{'Reju FID (lower best)':<25} | {v12_reju_fid:<20.2f} | {fid_res['reju_fid']:<20.2f} | {delta_reju_fid:<+20.2f}")
    print(f"{'Aging ACC (ResNet-18)':<25} | {v12_aging_acc:<19.2f}% | {cls2_res['aging_acc']:<19.2f}% | {delta_aging_acc:<+19.2f} pp")
    print(f"{'Reju ACC (ResNet-18)':<25} | {v12_reju_acc:<19.2f}% | {cls2_res['reju_acc']:<19.2f}% | {delta_reju_acc:<+19.2f} pp")
    print(f"{'Overall ACC':<25} | {v12_overall_acc:<19.2f}% | {cls2_res['overall_acc']:<19.2f}% | {delta_overall_acc:<+19.2f} pp")
    print(f"{'='*95}")


if __name__ == '__main__':
    main()
