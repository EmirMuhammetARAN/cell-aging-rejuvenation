"""CycleGAN v2: Generate all translations + evaluate (ACC + FID)."""
import os, sys, torch, gc
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from pytorch_fid import fid_score

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.cyclegan.generator_resnet import GeneratorResNet
from models.cyclegan.discriminator import Discriminator
from models.cyclegan.cyclegan_gan_model import CycleGANModel
from models.classifier.classifier import Classifier

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Paths
CHECKPOINT = os.path.join(root_dir, 'checkpoints', 'cyclegan_v2_epoch_110.pth')
TEST_YOUNG = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'young')
TEST_SENES = os.path.join(root_dir, 'data', 'processed_v2', 'test', 'senescent')
OUTPUT_DIR = os.path.join(root_dir, 'results', 'generated', 'cyclegan_v2')
os.makedirs(os.path.join(OUTPUT_DIR, 'aging'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'rejuvenation'), exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load model
model = CycleGANModel(GeneratorResNet, Discriminator, DEVICE, use_lpips=False)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.to(DEVICE, memory_format=torch.channels_last)
model.eval()
print(f"Loaded: {CHECKPOINT}")

# 1) Aging: Young -> Senescent (G_AB)
young_files = sorted(os.listdir(TEST_YOUNG))
print(f"\n[1/2] Aging: {len(young_files)} young -> senescent")
for img_name in tqdm(young_files, desc="Aging"):
    img = transform(Image.open(os.path.join(TEST_YOUNG, img_name)).convert('RGB'))
    img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            fake = model.G_AB(img)
    fake = (fake.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    transforms.ToPILImage()(fake).save(os.path.join(OUTPUT_DIR, 'aging', img_name))

# 2) Rejuvenation: Senescent -> Young (G_BA)
senes_files = sorted(os.listdir(TEST_SENES))
print(f"\n[2/2] Rejuvenation: {len(senes_files)} senescent -> young")
for img_name in tqdm(senes_files, desc="Rejuvenation"):
    img = transform(Image.open(os.path.join(TEST_SENES, img_name)).convert('RGB'))
    img = img.unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            fake = model.G_BA(img)
    fake = (fake.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
    transforms.ToPILImage()(fake).save(os.path.join(OUTPUT_DIR, 'rejuvenation', img_name))

del model; torch.cuda.empty_cache(); gc.collect()

# ===== EVALUATE =====
print(f"\n{'='*60}")
print("EVALUATION")
print(f"{'='*60}")

# Classifier
classifier = Classifier(output_size=2)
classifier.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')))
classifier.to(DEVICE); classifier.eval()

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tasks = [
    ('aging', 0, TEST_SENES),
    ('rejuvenation', 1, TEST_YOUNG),
]

for folder, expected_class, real_dir in tasks:
    folder_path = os.path.join(OUTPUT_DIR, folder)
    correct = total = 0
    for img_name in os.listdir(folder_path):
        with torch.no_grad():
            img = cls_transform(Image.open(os.path.join(folder_path, img_name)).convert('RGB'))
            pred = classifier(img.unsqueeze(0).to(DEVICE)).argmax(dim=1).item()
            total += 1
            if pred == expected_class: correct += 1
    acc = correct / total * 100

    fid = fid_score.calculate_fid_given_paths(
        [real_dir, folder_path], batch_size=32, device=DEVICE, dims=2048, num_workers=0)

    print(f"{folder}: ACC={acc:.2f}% ({correct}/{total}) | FID={fid:.2f}")

print(f"{'='*60}")
