"""
Parameter sweep for pretrained VAE model: translate full test set with given params, then evaluate.
Usage: python param_sweep.py --aging_strength 0.7 --aging_cfg 4.0 --reju_strength 0.65 --reju_cfg 3.0
"""
import os, sys, argparse, torch, gc
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
from models.classifier.classifier import Classifier

parser = argparse.ArgumentParser()
parser.add_argument('--aging_strength', type=float, default=0.8)
parser.add_argument('--aging_cfg', type=float, default=5.0)
parser.add_argument('--reju_strength', type=float, default=0.7)
parser.add_argument('--reju_cfg', type=float, default=4.0)
parser.add_argument('--steps', type=int, default=50)
args = parser.parse_args()

tag = f"as{args.aging_strength}_ac{args.aging_cfg}_rs{args.reju_strength}_rc{args.reju_cfg}"
print(f"\n{'='*60}")
print(f"SWEEP: {tag}")
print(f"Aging: strength={args.aging_strength}, CFG={args.aging_cfg}")
print(f"Reju:  strength={args.reju_strength}, CFG={args.reju_cfg}")
print(f"{'='*60}\n")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', f'sweep_{tag}')
os.makedirs(os.path.join(OUTPUT_DIR, 'aging'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejuvenation'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load LDM (pretrained VAE, no custom vae_path)
model = CellLDM(num_classes=2, lpips_weight=0.0)
model.scaling_factor = 0.18215
model.to(DEVICE, memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)
model.init_ema()

ckpt = torch.load(CHECKPOINT, map_location='cpu')

# Handle older v1 checkpoints that have 2 class embeddings instead of 3
for key in ['unet_state_dict', 'ema_unet_state_dict']:
    if key in ckpt and 'class_embedding.weight' in ckpt[key]:
        weight = ckpt[key]['class_embedding.weight']
        if weight.shape[0] == 2:
            # Pad with a 3rd zero embedding for the null class (CFG)
            pad_weight = torch.zeros(1, weight.shape[1], device=weight.device)
            ckpt[key]['class_embedding.weight'] = torch.cat([weight, pad_weight], dim=0)

model.unet.load_state_dict(ckpt['unet_state_dict'])
if 'ema_unet_state_dict' in ckpt:
    model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
print(f"Loaded checkpoint, epoch={ckpt.get('epoch','?')}")
del ckpt; gc.collect()
model.eval()

# === AGING (young -> senescent) ===
print(f"\n[1/2] Aging: {len(os.listdir(TEST_YOUNG))} images")
skipped_aging = 0
for img_name in tqdm(sorted(os.listdir(TEST_YOUNG)), desc="Aging"):
    out_path = os.path.join(OUTPUT_DIR, 'aging', img_name)
    if os.path.exists(out_path):
        skipped_aging += 1
        continue
    img = transform(Image.open(os.path.join(TEST_YOUNG, img_name)).convert('RGB'))
    img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    with torch.no_grad():
        out = model.translate(img, target_labels=torch.tensor([1], device=DEVICE),
                              strength=args.aging_strength, num_steps=args.steps,
                              use_ema=True, guidance_scale=args.aging_cfg)
    save_image(out.cpu(), out_path)
    if DEVICE == 'cuda': torch.cuda.empty_cache()
if skipped_aging: print(f"  Skipped {skipped_aging} existing aging images")

# === REJUVENATION (senescent -> young) ===
print(f"\n[2/2] Rejuvenation: {len(os.listdir(TEST_SENES))} images")
skipped_reju = 0
for img_name in tqdm(sorted(os.listdir(TEST_SENES)), desc="Reju"):
    out_path = os.path.join(OUTPUT_DIR, 'rejuvenation', img_name)
    if os.path.exists(out_path):
        skipped_reju += 1
        continue
    img = transform(Image.open(os.path.join(TEST_SENES, img_name)).convert('RGB'))
    img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    with torch.no_grad():
        out = model.translate(img, target_labels=torch.tensor([0], device=DEVICE),
                              strength=args.reju_strength, num_steps=args.steps,
                              use_ema=True, guidance_scale=args.reju_cfg)
    save_image(out.cpu(), out_path)
    if DEVICE == 'cuda': torch.cuda.empty_cache()
if skipped_reju: print(f"  Skipped {skipped_reju} existing rejuvenation images")

del model
if DEVICE == 'cuda': torch.cuda.empty_cache()
gc.collect()

# === EVALUATE ===
print(f"\n{'='*60}")
print("EVALUATION")
print(f"{'='*60}")

# Classifier
classifier = Classifier(output_size=2)
classifier.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth'), map_location=DEVICE))
classifier.to(DEVICE); classifier.eval()

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

tasks = [
    ('aging', 0, TEST_SENES),
    ('rejuvenation', 1, TEST_YOUNG),
]

results = {}
for folder, expected_class, real_dir in tasks:
    folder_path = os.path.join(OUTPUT_DIR, folder)
    correct = total = 0
    for img_name in os.listdir(folder_path):
        with torch.no_grad():
            img = cls_transform(Image.open(os.path.join(folder_path, img_name)).convert('RGB'))
            pred = classifier(img.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()
            total += 1
            if pred == expected_class: correct += 1
    acc = correct/total*100 if total > 0 else 0
    
    # FID
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, folder_path], batch_size=32, device=DEVICE, dims=2048, num_workers=0)
    
    results[folder] = {'acc': acc, 'fid': fid, 'correct': correct, 'total': total}
    print(f"{folder}: ACC={acc:.2f}% ({correct}/{total}) | FID={fid:.2f}")

del classifier
if DEVICE == 'cuda': torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: {tag}")
print(f"Aging:  strength={args.aging_strength} CFG={args.aging_cfg} -> ACC={results['aging']['acc']:.2f}% FID={results['aging']['fid']:.2f}")
print(f"Reju:   strength={args.reju_strength} CFG={args.reju_cfg} -> ACC={results['rejuvenation']['acc']:.2f}% FID={results['rejuvenation']['fid']:.2f}")
print(f"{'='*60}")
