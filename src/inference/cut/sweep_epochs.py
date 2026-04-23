"""CUT: Sweep all epoch checkpoints and evaluate using ACC and FID."""
import os, sys, torch, gc
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.cut.generator import CUTGenerator
from models.classifier.classifier import Classifier

torch.backends.cudnn.benchmark = True
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')
CHECKPOINT_DIR = os.path.join(root_dir, 'checkpoints', 'cut')
OUTPUT_BASE = os.path.join(root_dir, 'results', 'generated', 'cut_sweep')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Find all CUT epoch checkpoints
epochs = sorted([
    int(f.replace('cut_epoch_', '').replace('.pth', ''))
    for f in os.listdir(CHECKPOINT_DIR)
    if f.startswith('cut_epoch_') and f.endswith('.pth')
])

print(f"Found checkpoints for epochs: {epochs}")
print(f"{'='*70}")

# Load classifier once
classifier = Classifier(output_size=2)
classifier.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth'), weights_only=True))
classifier.to(DEVICE); classifier.eval()

young_files = sorted(os.listdir(TEST_YOUNG))
senes_files = sorted(os.listdir(TEST_SENES))

results = []

for epoch in epochs:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'cut_epoch_{epoch}.pth')
    out_dir = os.path.join(OUTPUT_BASE, f'epoch_{epoch}')
    os.makedirs(os.path.join(out_dir, 'aging'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'rejuvenation'), exist_ok=True)

    # Load CUT Generators
    aging_G = CUTGenerator(num_residual_blocks=9)
    reju_G = CUTGenerator(num_residual_blocks=9)
    
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    aging_G.load_state_dict(ckpt['aging_G'])
    reju_G.load_state_dict(ckpt['reju_G'])
    
    aging_G.to(DEVICE, memory_format=torch.channels_last).eval()
    reju_G.to(DEVICE, memory_format=torch.channels_last).eval()

    # Generate aging
    for img_name in tqdm(young_files, desc=f"Ep{epoch} aging", leave=False):
        img = transform(Image.open(os.path.join(TEST_YOUNG, img_name)).convert('RGB'))
        img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                fake = aging_G(img)
        fake = (fake.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        transforms.ToPILImage()(fake).save(os.path.join(out_dir, 'aging', img_name))

    # Generate rejuvenation
    for img_name in tqdm(senes_files, desc=f"Ep{epoch} reju", leave=False):
        img = transform(Image.open(os.path.join(TEST_SENES, img_name)).convert('RGB'))
        img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                fake = reju_G(img)
        fake = (fake.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        transforms.ToPILImage()(fake).save(os.path.join(out_dir, 'rejuvenation', img_name))

    del aging_G, reju_G, ckpt; torch.cuda.empty_cache(); gc.collect()

    # Evaluate
    # Aging ACC
    correct_a = total_a = 0
    for img_name in os.listdir(os.path.join(out_dir, 'aging')):
        with torch.no_grad():
            img = cls_transform(Image.open(os.path.join(out_dir, 'aging', img_name)).convert('RGB'))
            pred = classifier(img.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()
            total_a += 1
            if pred == 0: correct_a += 1
    acc_a = correct_a / total_a * 100

    # Reju ACC
    correct_r = total_r = 0
    for img_name in os.listdir(os.path.join(out_dir, 'rejuvenation')):
        with torch.no_grad():
            img = cls_transform(Image.open(os.path.join(out_dir, 'rejuvenation', img_name)).convert('RGB'))
            pred = classifier(img.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()
            total_r += 1
            if pred == 1: correct_r += 1
    acc_r = correct_r / total_r * 100

    # FID
    fid_a = fid_score.calculate_fid_given_paths(
        [TEST_SENES, os.path.join(out_dir, 'aging')], batch_size=32, device=DEVICE, dims=2048, num_workers=0)
    fid_r = fid_score.calculate_fid_given_paths(
        [TEST_YOUNG, os.path.join(out_dir, 'rejuvenation')], batch_size=32, device=DEVICE, dims=2048, num_workers=0)

    results.append((epoch, acc_a, fid_a, acc_r, fid_r))
    print(f"Epoch {epoch:>3d} | Aging: ACC={acc_a:5.2f}% FID={fid_a:5.2f} | Reju: ACC={acc_r:5.2f}% FID={fid_r:5.2f}")

print(f"\n{'='*70}")
print(f"{'Epoch':>5} | {'Aging ACC':>9} {'Aging FID':>9} | {'Reju ACC':>9} {'Reju FID':>9}")
print(f"{'-'*70}")
for ep, aa, af, ra, rf in results:
    print(f"{ep:>5} | {aa:>8.2f}% {af:>9.2f} | {ra:>8.2f}% {rf:>9.2f}")
