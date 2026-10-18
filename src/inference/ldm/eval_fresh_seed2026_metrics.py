"""
Calculate FID and ResNet-18 Accuracy for fresh LDM Seed 2026
"""
import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
OUT_DIR = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026')
AGING_DIR = os.path.join(OUT_DIR, 'aging')
REJUV_DIR = os.path.join(OUT_DIR, 'rejuv')
CLASSIFIER_CKPT = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')


def main():
    print("=" * 70)
    print(" EVALUATING FRESH LDM SEED 2026 METRICS (N=645) ")
    print("=" * 70)

    # 1. Classifier Evaluation
    cls_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classifier = Classifier(output_size=2).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=DEVICE))
    classifier.eval()

    def eval_classifier(folder, target_class):
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))])
        correct = 0
        confs = []
        for i in range(0, len(files), 32):
            batch_files = files[i:i+32]
            tensors = torch.stack([cls_transform(Image.open(os.path.join(folder, f)).convert('RGB')) for f in batch_files]).to(DEVICE)
            with torch.no_grad():
                logits = classifier(tensors)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                correct += (preds == target_class).sum().item()
                confs.extend(probs[:, target_class].cpu().numpy().tolist())
        acc = (correct / len(files)) * 100.0
        conf = float(np.mean(confs)) * 100.0
        return acc, conf

    # Class 0: Senescent, Class 1: Young
    aging_acc, aging_conf = eval_classifier(AGING_DIR, target_class=0)
    reju_acc, reju_conf = eval_classifier(REJUV_DIR, target_class=1)
    overall_acc = (aging_acc * 327 + reju_acc * 318) / 645
    overall_conf = (aging_conf * 327 + reju_conf * 318) / 645

    print(f"[ACC] Aging Transition Acc: {aging_acc:.2f}% (Conf: {aging_conf:.2f}%)")
    print(f"[ACC] Rejuv Transition Acc: {reju_acc:.2f}% (Conf: {reju_conf:.2f}%)")
    print(f"[ACC] Overall Acc:         {overall_acc:.2f}% (Conf: {overall_conf:.2f}%)")

    # 2. FID Evaluation
    print("\n[*] Calculating FID...")
    aging_fid = fid_score.calculate_fid_given_paths([TEST_SENES, AGING_DIR], batch_size=50, device=DEVICE, dims=2048)
    reju_fid = fid_score.calculate_fid_given_paths([TEST_YOUNG, REJUV_DIR], batch_size=50, device=DEVICE, dims=2048)

    print(f"[FID] Aging FID: {aging_fid:.2f}")
    print(f"[FID] Rejuv FID: {reju_fid:.2f}")

    summary = {
        'seed': 2026,
        'timestamp': '2026-10-14',
        'aging_fid': aging_fid,
        'aging_acc': aging_acc,
        'aging_conf': aging_conf,
        'reju_fid': reju_fid,
        'reju_acc': reju_acc,
        'reju_conf': reju_conf,
        'overall_acc': overall_acc,
        'overall_conf': overall_conf
    }

    out_json = os.path.join(OUT_DIR, 'fresh_seed_2026_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Metrics saved to {out_json}")


if __name__ == '__main__':
    main()
