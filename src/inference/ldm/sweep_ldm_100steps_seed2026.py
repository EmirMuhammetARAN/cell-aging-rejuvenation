"""
LDM v12 Checkpoint Sweep with 100 DDIM Steps & Fixed Seed 2026
==============================================================
Evaluates 7 milestone checkpoints on the frozen test cohort (N=645):
  - Parameters: Low-FID Regime (Aging str=0.75, cfg=4.0 | Rejuv str=0.65, cfg=3.5)
  - Sampling: 100 DDIM Steps
  - Seed: 2026 (strictly fixed across all checkpoints)

Checkpoints:
  1. Base Epoch 50
  2. Base Epoch 100
  3. Base Epoch 150
  4. Base Epoch 200
  5. Base Epoch 250
  6. LPIPS Epoch 25 (Ep 275)
  7. LPIPS Epoch 50 (Ep 300 / Last) -> Already computed in results/generated/seed_sweep/seed_2026
"""

import os
import sys
import gc
import json
import shutil
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
from models.classifier.classifier import Classifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True

FIXED_SEED = 2026
AGING_STRENGTH = 0.75
AGING_CFG = 4.0
REJUV_STRENGTH = 0.65
REJUV_CFG = 3.5
NUM_STEPS = 100
BATCH_SIZE = 8

TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CKPT_DIR = os.path.join(root_dir, 'checkpoints', 'ldm')
OUT_BASE = os.path.join(root_dir, 'results', 'generated', 'ldm_v12_sweep_100steps_seed2026')
os.makedirs(OUT_BASE, exist_ok=True)

transform_gen = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CHECKPOINTS = [
    {'name': 'Base Epoch 50', 'epoch': 50, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_epoch_50.pt')},
    {'name': 'Base Epoch 100', 'epoch': 100, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_epoch_100.pt')},
    {'name': 'Base Epoch 150', 'epoch': 150, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_epoch_150.pt')},
    {'name': 'Base Epoch 200', 'epoch': 200, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_epoch_200.pt')},
    {'name': 'Base Epoch 250', 'epoch': 250, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_epoch_250.pt')},
    {'name': 'Latent-L1 FT (Total Epoch 275)', 'epoch': 275, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_lpips_epoch_25.pt')},
    {'name': 'Latent-L1 FT (Total Epoch 300)', 'epoch': 300, 'ckpt': os.path.join(CKPT_DIR, 'checkpoint_v12_v4_data_lpips_last.pt')}
]


def set_seed(seed=FIXED_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model(ckpt_path):
    model = CellLDM(num_classes=2).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location='cpu')
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
    return model


def main():
    print("=" * 80)
    print(" LDM v12 BENCHMARK: 100 DDIM STEPS, LOW-FID REGIME, FIXED SEED 2026 ")
    print("=" * 80)

    # 1. Check if Epoch 300 can be imported from seed_sweep/seed_2026
    seed2026_src = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026')
    ep300_dst = os.path.join(OUT_BASE, 'epoch_300')
    if os.path.exists(seed2026_src) and not os.path.exists(ep300_dst):
        print("[*] Copying precomputed Epoch 300 (seed_2026, 100 steps) to sweep folder...")
        shutil.copytree(seed2026_src, ep300_dst)
        print("[OK] Epoch 300 ready.")

    young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.lower().endswith(('.jpg', '.png'))])
    senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.lower().endswith(('.jpg', '.png'))])

    # 2. Iterate through all 7 checkpoints
    for c_info in CHECKPOINTS:
        epoch = c_info['epoch']
        name = c_info['name']
        ckpt_p = c_info['ckpt']

        ep_out = os.path.join(OUT_BASE, f'epoch_{epoch}')
        aging_dir = os.path.join(ep_out, 'aging')
        reju_dir = os.path.join(ep_out, 'rejuv')
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(reju_dir, exist_ok=True)

        existing_a = len(os.listdir(aging_dir))
        existing_r = len(os.listdir(reju_dir))

        if existing_a >= len(young_files) and existing_r >= len(senes_files):
            print(f"[OK] Epoch {epoch} already fully generated ({existing_a} aging, {existing_r} rejuv). Skipping generation.")
            continue

        print(f"\n{'='*70}")
        print(f"Generating Epoch {epoch} ({name}) with 100 DDIM Steps & Seed {FIXED_SEED}")
        print(f"{'='*70}")

        model = load_model(ckpt_p)

        # 2a. Aging
        set_seed(FIXED_SEED)
        for i in tqdm(range(0, len(young_files), BATCH_SIZE), desc=f"Ep{epoch} Aging"):
            b_files = young_files[i:i+BATCH_SIZE]
            pending = [f for f in b_files if not os.path.exists(os.path.join(aging_dir, f))]
            if not pending:
                continue
            imgs = [transform_gen(Image.open(os.path.join(TEST_YOUNG, f)).convert('RGB')) for f in pending]
            b_tensor = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    fakes = model.translate(
                        b_tensor,
                        target_labels=torch.tensor([1] * len(pending), device=DEVICE),
                        strength=AGING_STRENGTH,
                        num_steps=NUM_STEPS,
                        guidance_scale=AGING_CFG,
                        use_ema=True
                    )
            for f, fake in zip(pending, fakes):
                transforms.ToPILImage()(fake.float().cpu().clamp(0, 1)).save(os.path.join(aging_dir, f))

        # 2b. Rejuvenation
        set_seed(FIXED_SEED)
        for i in tqdm(range(0, len(senes_files), BATCH_SIZE), desc=f"Ep{epoch} Rejuv"):
            b_files = senes_files[i:i+BATCH_SIZE]
            pending = [f for f in b_files if not os.path.exists(os.path.join(reju_dir, f))]
            if not pending:
                continue
            imgs = [transform_gen(Image.open(os.path.join(TEST_SENES, f)).convert('RGB')) for f in pending]
            b_tensor = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    fakes = model.translate(
                        b_tensor,
                        target_labels=torch.tensor([0] * len(pending), device=DEVICE),
                        strength=REJUV_STRENGTH,
                        num_steps=NUM_STEPS,
                        guidance_scale=REJUV_CFG,
                        use_ema=True
                    )
            for f, fake in zip(pending, fakes):
                transforms.ToPILImage()(fake.float().cpu().clamp(0, 1)).save(os.path.join(reju_dir, f))

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 3. Metric Evaluations
    print(f"\n{'='*75}")
    print("COMPUTING FINAL RESNET-18 ACC & FID FOR ALL 7 CHECKPOINTS (100 STEPS, SEED 2026)")
    print(f"{'='*75}")

    classifier = Classifier(output_size=2).to(DEVICE)
    cls_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier.load_state_dict(torch.load(cls_path, map_location=DEVICE))
    classifier.eval()

    results = []

    for c_info in CHECKPOINTS:
        epoch = c_info['epoch']
        name = c_info['name']
        ep_out = os.path.join(OUT_BASE, f'epoch_{epoch}')
        aging_dir = os.path.join(ep_out, 'aging')
        reju_dir = os.path.join(ep_out, 'rejuv')

        # Aging ACC (Target = 0 senescent)
        correct_a = 0
        confs_a = []
        for f in young_files:
            img = cls_transform(Image.open(os.path.join(aging_dir, f)).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(classifier(img), dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs[0, 0].item()
                confs_a.append(conf)
                if pred == 0:
                    correct_a += 1
        acc_a = (correct_a / len(young_files)) * 100.0
        conf_a = (sum(confs_a) / len(confs_a)) * 100.0

        # Rejuv ACC (Target = 1 young)
        correct_r = 0
        confs_r = []
        for f in senes_files:
            img = cls_transform(Image.open(os.path.join(reju_dir, f)).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(classifier(img), dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs[0, 1].item()
                confs_r.append(conf)
                if pred == 1:
                    correct_r += 1
        acc_r = (correct_r / len(senes_files)) * 100.0
        conf_r = (sum(confs_r) / len(confs_r)) * 100.0

        tot_cells = len(young_files) + len(senes_files)
        tot_correct = correct_a + correct_r
        overall_acc = (tot_correct / tot_cells) * 100.0
        overall_conf = ((sum(confs_a) + sum(confs_r)) / tot_cells) * 100.0

        # FID
        fid_a = fid_score.calculate_fid_given_paths([TEST_SENES, aging_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)
        fid_r = fid_score.calculate_fid_given_paths([TEST_YOUNG, reju_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0)

        rec = {
            'checkpoint_name': name,
            'epoch': epoch,
            'aging_acc': acc_a,
            'aging_conf': conf_a,
            'aging_fid': fid_a,
            'reju_acc': acc_r,
            'reju_conf': conf_r,
            'reju_fid': fid_r,
            'overall_acc': overall_acc,
            'overall_conf': overall_conf
        }
        results.append(rec)
        print(f"Epoch {epoch:>3d} | Aging: ACC={acc_a:5.2f}% FID={fid_a:5.2f} | Reju: ACC={acc_r:5.2f}% FID={fid_r:5.2f} | Overall: ACC={overall_acc:5.2f}%")

    json_path = os.path.join(OUT_BASE, 'ldm_v12_sweep_100steps_seed2026_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_BASE, 'ldm_v12_sweep_100steps_seed2026_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Metrics saved to {json_path} and {csv_path}")

    print("\n" + "="*95)
    print(f"{'Checkpoint':<28} | {'Epoch':>5} | {'Aging ACC':>9} {'Aging FID':>9} | {'Reju ACC':>9} {'Reju FID':>9} | {'Overall ACC':>11}")
    print("-"*95)
    for r in results:
        print(f"{r['checkpoint_name']:<28} | {r['epoch']:>5} | {r['aging_acc']:>8.2f}% {r['aging_fid']:>9.2f} | {r['reju_acc']:>8.2f}% {r['reju_fid']:>9.2f} | {r['overall_acc']:>10.2f}%")
    print("="*95)


if __name__ == '__main__':
    main()
