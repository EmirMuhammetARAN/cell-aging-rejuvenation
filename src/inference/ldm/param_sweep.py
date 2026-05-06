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

CHECKPOINT = os.path.join(root_dir, 'checkpoints', 'ldm', 'best_model_v12_v3_data.pt')
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v3', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v3', 'test', 'senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'ldm', f'sweep_{tag}')
os.makedirs(os.path.join(OUTPUT_DIR, 'aging'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejuvenation'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

# Load LDM (pretrained VAE, no custom vae_path)
model = CellLDM(num_classes=2, lpips_weight=0.0)
model.scaling_factor = 0.18215
model.to('cuda', memory_format=torch.channels_last)
model.vae.to(memory_format=torch.channels_last)
model.init_ema()

ckpt = torch.load(CHECKPOINT, map_location='cpu')
model.unet.load_state_dict(ckpt['unet_state_dict'])
if 'ema_unet_state_dict' in ckpt:
    model.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
print(f"Loaded checkpoint, epoch={ckpt.get('epoch','?')}")
del ckpt; gc.collect()
model.eval()

# === AGING (young -> senescent) ===
print(f"\n[1/2] Aging: {len(os.listdir(TEST_YOUNG))} images")
for img_name in tqdm(sorted(os.listdir(TEST_YOUNG)), desc="Aging"):
    img = transform(Image.open(os.path.join(TEST_YOUNG, img_name)).convert('RGB'))
    img = img.unsqueeze(0).to('cuda', memory_format=torch.channels_last)
    with torch.no_grad():
        out = model.translate(img, target_labels=torch.tensor([1], device='cuda'),
                              strength=args.aging_strength, num_steps=args.steps,
                              use_ema=True, guidance_scale=args.aging_cfg)
    save_image(out.cpu(), os.path.join(OUTPUT_DIR, 'aging', img_name))
    torch.cuda.empty_cache()

# === REJUVENATION (senescent -> young) ===
print(f"\n[2/2] Rejuvenation: {len(os.listdir(TEST_SENES))} images")
for img_name in tqdm(sorted(os.listdir(TEST_SENES)), desc="Reju"):
    img = transform(Image.open(os.path.join(TEST_SENES, img_name)).convert('RGB'))
    img = img.unsqueeze(0).to('cuda', memory_format=torch.channels_last)
    with torch.no_grad():
        out = model.translate(img, target_labels=torch.tensor([0], device='cuda'),
                              strength=args.reju_strength, num_steps=args.steps,
                              use_ema=True, guidance_scale=args.reju_cfg)
    save_image(out.cpu(), os.path.join(OUTPUT_DIR, 'rejuvenation', img_name))
    torch.cuda.empty_cache()

del model; torch.cuda.empty_cache(); gc.collect()

# === EVALUATE ===
print(f"\n{'='*60}")
print("EVALUATION")
print(f"{'='*60}")

# Classifier
classifier = Classifier(output_size=2)
classifier.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')))
classifier.to('cuda'); classifier.eval()

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
            pred = classifier(img.unsqueeze(0).to('cuda')).argmax(dim=1).item()
            total += 1
            if pred == expected_class: correct += 1
    acc = correct/total*100 if total > 0 else 0
    
    # FID
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, folder_path], batch_size=32, device='cuda', dims=2048, num_workers=0)
    
    results[folder] = {'acc': acc, 'fid': fid, 'correct': correct, 'total': total}
    print(f"{folder}: ACC={acc:.2f}% ({correct}/{total}) | FID={fid:.2f}")

del classifier; torch.cuda.empty_cache()

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY: {tag}")
print(f"Aging:  strength={args.aging_strength} CFG={args.aging_cfg} -> ACC={results['aging']['acc']:.2f}% FID={results['aging']['fid']:.2f}")
print(f"Reju:   strength={args.reju_strength} CFG={args.reju_cfg} -> ACC={results['rejuvenation']['acc']:.2f}% FID={results['rejuvenation']['fid']:.2f}")
print(f"{'='*60}")
