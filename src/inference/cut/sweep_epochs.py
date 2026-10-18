"""
CUT GAN: Sweep All Epoch Checkpoints and Evaluate Phenotypic Transition & Realism
===================================================================================
Iterates through all CUT GAN training epoch checkpoints (Epoch 10 to 200),
generates bidirectional single-cell translations on the frozen held-out test cohort (N=645):
  - Aging: 327 young cells -> senescent
  - Rejuvenation: 318 senescent cells -> young

Computes:
  - ResNet-18 Classifier Transition Accuracy (%) & Target Confidence (%)
  - Fréchet Inception Distance (FID) vs. ground-truth test reference sets
  - Overall Transition Accuracy (%) across both directions

Saves full metrics to results/generated/cut_sweep/cut_sweep_metrics.json & .csv.
"""

import os
import sys
import gc
import json
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.cut.generator import CUTGenerator
from models.classifier.classifier import Classifier

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints', 'cut')
OUTPUT_BASE = os.path.join(root_dir, 'results', 'generated', 'cut_sweep')
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Transforms
transform_gen = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def main():
    # Identify standard 10-step checkpoints (10 to 200)
    all_ckpts = [
        int(f.replace('cut_epoch_', '').replace('.pth', ''))
        for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith('cut_epoch_') and f.endswith('.pth') and not '_' in f.replace('cut_epoch_', '').replace('.pth', '')
    ]
    epochs = sorted([e for e in all_ckpts if e % 10 == 0])

    print("=" * 80)
    print("           CUT GAN EPOCH SWEEP BENCHMARK (EPOCHS 10 TO 200)            ")
    print("=" * 80)
    print(f"Discovered {len(epochs)} checkpoints: {epochs}")
    print(f"Test cohort: Aging = {len(os.listdir(TEST_YOUNG))} cells, Rejuvenation = {len(os.listdir(TEST_SENES))} cells")

    # Load Classifier v2 once
    classifier = Classifier(output_size=2).to(DEVICE)
    cls_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier.load_state_dict(torch.load(cls_path, map_location=DEVICE))
    classifier.eval()
    print("[OK] Loaded ResNet-18 Classifier.")

    young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.lower().endswith(('.jpg', '.png'))])
    senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.lower().endswith(('.jpg', '.png'))])

    results = []

    for epoch in epochs:
        print(f"\n{'='*70}")
        print(f"Evaluating CUT GAN Epoch {epoch}")
        print(f"{'='*70}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f'cut_epoch_{epoch}.pth')
        out_dir = os.path.join(OUTPUT_BASE, f'epoch_{epoch}')
        aging_dir = os.path.join(out_dir, 'aging')
        reju_dir = os.path.join(out_dir, 'rejuvenation')
        os.makedirs(aging_dir, exist_ok=True)
        os.makedirs(reju_dir, exist_ok=True)

        # 1. Load Generator Checkpoint
        aging_G = CUTGenerator(num_residual_blocks=9)
        reju_G = CUTGenerator(num_residual_blocks=9)

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        aging_G.load_state_dict(ckpt['aging_G'])
        reju_G.load_state_dict(ckpt['reju_G'])

        aging_G.to(DEVICE, memory_format=torch.channels_last).eval()
        reju_G.to(DEVICE, memory_format=torch.channels_last).eval()

        # 2. Generate Aging (Young -> Senescent)
        for img_name in tqdm(young_files, desc=f"Ep{epoch} aging", leave=False):
            out_p = os.path.join(aging_dir, img_name)
            if not os.path.exists(out_p):
                img = transform_gen(Image.open(os.path.join(TEST_YOUNG, img_name)).convert('RGB'))
                img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        fake = aging_G(img)
                fake = (fake.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
                transforms.ToPILImage()(fake).save(out_p)

        # 3. Generate Rejuvenation (Senescent -> Young)
        for img_name in tqdm(senes_files, desc=f"Ep{epoch} reju", leave=False):
            out_p = os.path.join(reju_dir, img_name)
            if not os.path.exists(out_p):
                img = transform_gen(Image.open(os.path.join(TEST_SENES, img_name)).convert('RGB'))
                img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        fake = reju_G(img)
                fake = (fake.squeeze(0).cpu().float() * 0.5 + 0.5).clamp(0, 1)
                transforms.ToPILImage()(fake).save(out_p)

        del aging_G, reju_G, ckpt
        torch.cuda.empty_cache()
        gc.collect()

        # 4. Evaluate Classifier v2 on Aging (target = 0 senescent)
        correct_a = 0
        confs_a = []
        for img_name in young_files:
            img = cls_transform(Image.open(os.path.join(aging_dir, img_name)).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(classifier(img), dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs[0, 0].item()
                confs_a.append(conf)
                if pred == 0:
                    correct_a += 1
        acc_a = (correct_a / len(young_files)) * 100.0
        conf_a = (sum(confs_a) / len(confs_a)) * 100.0

        # 5. Evaluate Classifier v2 on Rejuvenation (target = 1 young)
        correct_r = 0
        confs_r = []
        for img_name in senes_files:
            img = cls_transform(Image.open(os.path.join(reju_dir, img_name)).convert('RGB')).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probs = torch.softmax(classifier(img), dim=1)
                pred = probs.argmax(dim=1).item()
                conf = probs[0, 1].item()
                confs_r.append(conf)
                if pred == 1:
                    correct_r += 1
        acc_r = (correct_r / len(senes_files)) * 100.0
        conf_r = (sum(confs_r) / len(confs_r)) * 100.0

        total_cells = len(young_files) + len(senes_files)
        total_correct = correct_a + correct_r
        overall_acc = (total_correct / total_cells) * 100.0
        overall_conf = ((sum(confs_a) + sum(confs_r)) / total_cells) * 100.0

        # 6. Evaluate FID
        fid_a = fid_score.calculate_fid_given_paths(
            [TEST_SENES, aging_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0
        )
        fid_r = fid_score.calculate_fid_given_paths(
            [TEST_YOUNG, reju_dir], batch_size=32, device=DEVICE, dims=2048, num_workers=0
        )

        rec = {
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

        print(f"Epoch {epoch:>3d} | Aging: ACC={acc_a:5.2f}% FID={fid_a:5.2f} | Reju: ACC={acc_r:5.2f}% FID={fid_r:5.2f} | Overall ACC={overall_acc:5.2f}%")

    # Save summary files
    json_path = os.path.join(OUTPUT_BASE, 'cut_sweep_metrics.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved results JSON to {json_path}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_BASE, 'cut_sweep_metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"[OK] Saved results CSV to {csv_path}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Epoch':>5} | {'Aging ACC':>9} {'Aging FID':>9} | {'Reju ACC':>9} {'Reju FID':>9} | {'Overall ACC':>11}")
    print(f"{'-'*90}")
    for r in results:
        print(f"{r['epoch']:>5} | {r['aging_acc']:>8.2f}% {r['aging_fid']:>9.2f} | {r['reju_acc']:>8.2f}% {r['reju_fid']:>9.2f} | {r['overall_acc']:>10.2f}%")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()
