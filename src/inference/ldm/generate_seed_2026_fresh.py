"""
Fresh Regeneration of LDM v12 Seed 2026 (Full FP32, 100 DDIM Steps, Low-FID Regime)
Outputs to: results/generated/seed_sweep/seed_2026/aging and rejuv
Then recalculates FID and Accuracy metrics.
"""

import os
import sys
import gc
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
from models.classifier.classifier import Classifier

# Environment / Precision
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

FIXED_SEED = 2026
AGING_STRENGTH = 0.75
AGING_CFG = 4.0
REJUV_STRENGTH = 0.65
REJUV_CFG = 3.5
NUM_STEPS = 100
BATCH_SIZE = 8

TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CKPT_PATH = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
OUT_DIR = os.path.join(root_dir, 'results', 'generated', 'seed_sweep', 'seed_2026')
AGING_DIR = os.path.join(OUT_DIR, 'aging')
REJUV_DIR = os.path.join(OUT_DIR, 'rejuv')
CLASSIFIER_CKPT = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')

os.makedirs(AGING_DIR, exist_ok=True)
os.makedirs(REJUV_DIR, exist_ok=True)


def set_seed(seed=FIXED_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def load_model():
    print(f"[*] Loading LDM Checkpoint: {CKPT_PATH}")
    model = CellLDM(num_classes=2).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
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
    print("=" * 75)
    print(f" FRESH LDM v12 GENERATION (SEED {FIXED_SEED}, FP32, 100 DDIM STEPS) ")
    print("=" * 75)

    transform_gen = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    young_files = sorted([f for f in os.listdir(TEST_YOUNG) if f.lower().endswith(('.jpg', '.png'))])
    senes_files = sorted([f for f in os.listdir(TEST_SENES) if f.lower().endswith(('.jpg', '.png'))])

    model = load_model()

    # 1. Aging (Young -> Senescent, target=1)
    print(f"\n[*] Generating Aging ({len(young_files)} cells, str={AGING_STRENGTH}, cfg={AGING_CFG})...")
    set_seed(FIXED_SEED)
    for i in tqdm(range(0, len(young_files), BATCH_SIZE), desc="Aging (Seed 2026)"):
        b_files = young_files[i:i+BATCH_SIZE]
        imgs = [transform_gen(Image.open(os.path.join(TEST_YOUNG, f)).convert('RGB')) for f in b_files]
        b_tensor = torch.stack(imgs).to(DEVICE)
        
        with torch.no_grad():
            fakes = model.translate(
                b_tensor,
                target_labels=torch.tensor([1] * len(b_files), device=DEVICE),
                strength=AGING_STRENGTH,
                num_steps=NUM_STEPS,
                guidance_scale=AGING_CFG,
                use_ema=True
            )
        for f, fake in zip(b_files, fakes):
            transforms.ToPILImage()(fake.float().cpu().clamp(0, 1)).save(os.path.join(AGING_DIR, f))

    # 2. Rejuvenation (Senescent -> Young, target=0)
    print(f"\n[*] Generating Rejuvenation ({len(senes_files)} cells, str={REJUV_STRENGTH}, cfg={REJUV_CFG})...")
    set_seed(FIXED_SEED)
    for i in tqdm(range(0, len(senes_files), BATCH_SIZE), desc="Rejuvenation (Seed 2026)"):
        b_files = senes_files[i:i+BATCH_SIZE]
        imgs = [transform_gen(Image.open(os.path.join(TEST_SENES, f)).convert('RGB')) for f in b_files]
        b_tensor = torch.stack(imgs).to(DEVICE)
        
        with torch.no_grad():
            fakes = model.translate(
                b_tensor,
                target_labels=torch.tensor([0] * len(b_files), device=DEVICE),
                strength=REJUV_STRENGTH,
                num_steps=NUM_STEPS,
                guidance_scale=REJUV_CFG,
                use_ema=True
            )
        for f, fake in zip(b_files, fakes):
            transforms.ToPILImage()(fake.float().cpu().clamp(0, 1)).save(os.path.join(REJUV_DIR, f))

    del model
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Calculate Metrics
    print("\n" + "=" * 75)
    print("[*] Calculating FID and Classifier Transition Accuracy...")
    print("=" * 75)

    cls_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    classifier = Classifier(num_classes=2).to(DEVICE)
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

    # Note: in classifier_v2, label 0 = senescent, label 1 = young
    # For aging, target is Senescent (0). For rejuv, target is Young (1).
    aging_acc, aging_conf = eval_classifier(AGING_DIR, target_class=0)
    reju_acc, reju_conf = eval_classifier(REJUV_DIR, target_class=1)

    print(f"Aging ResNet-18 Accuracy: {aging_acc:.2f}% (Avg Target Conf: {aging_conf:.2f}%)")
    print(f"Rejuv ResNet-18 Accuracy: {reju_acc:.2f}% (Avg Target Conf: {reju_conf:.2f}%)")

    # Calculate FID
    print("[*] Computing Fréchet Inception Distance (FID)...")
    aging_fid = fid_score.calculate_fid_given_paths([TEST_SENES, AGING_DIR], batch_size=50, device=DEVICE, dims=2048)
    reju_fid = fid_score.calculate_fid_given_paths([TEST_YOUNG, REJUV_DIR], batch_size=50, device=DEVICE, dims=2048)

    print(f"Aging FID: {aging_fid:.2f}")
    print(f"Rejuv FID: {reju_fid:.2f}")

    summary = {
        'seed': FIXED_SEED,
        'aging_fid': aging_fid,
        'aging_acc': aging_acc,
        'aging_conf': aging_conf,
        'reju_fid': reju_fid,
        'reju_acc': reju_acc,
        'reju_conf': reju_conf
    }

    out_json = os.path.join(OUT_DIR, 'fresh_seed_2026_metrics.json')
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Saved fresh metrics to: {out_json}")


if __name__ == '__main__':
    main()
