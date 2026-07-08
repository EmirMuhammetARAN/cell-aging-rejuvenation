import os
import sys
import subprocess
import torch
import torchvision.transforms as transforms
from PIL import Image
import lpips
from tqdm import tqdm

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)
from models.classifier.classifier import Classifier

RESULTS_DIR = os.path.join(root_dir, 'results', 'generated', 'checkpoint_comparison_lpips')
TEST_YOUNG_DIR = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
TEST_SENESCENT_DIR = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
CLASSIFIER_CKPT = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Classifier Transform (ResNet18 için standart ImageNet normalizasyonu kullanılmış olabilir, 
# train_classifier'a uygun olanı kullanıyoruz. LDM -1, 1 arasıydı, Classifier genelde 0-1 üstüne mean/std yapar.
# Ama projede genelde 0.5 mean/std kullanılmıştı. Kontrol ediyoruz.)
classifier_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Classifier'ı yükle
classifier_model = Classifier(output_size=2)
if os.path.exists(CLASSIFIER_CKPT):
    classifier_model.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location=device))
    classifier_model.to(device)
    classifier_model.eval()
    print("✓ Classifier model loaded for Evaluation.")
else:
    print(f"UYARI: Classifier checkpoint bulunamadı! Yol: {CLASSIFIER_CKPT}")
    classifier_model = None

def calculate_fid(path_real, path_fake):
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytorch_fid", path_real, path_fake, "--device", "cuda"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.split('\n'):
            if "FID:" in line:
                return float(line.split("FID:")[-1].strip())
    except Exception as e:
        print(f"FID Error for {path_fake}: {e}")
    return -1.0

def calculate_classifier_score(img_dir, target_label):
    """
    Belirli bir klasördeki resimleri okur ve classifier'ın bu resimlerin yüzde kaçını 
    target_label (istenen sınıf) olarak tahmin ettiğini döndürür.
    target_label = 1 (Aging için), target_label = 0 (Rejuv için)
    """
    if classifier_model is None:
        return 0.0
        
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if not img_names:
        return 0.0
        
    correct = 0
    total = len(img_names)
    
    with torch.no_grad():
        for img_name in img_names:
            img_path = os.path.join(img_dir, img_name)
            img = Image.open(img_path).convert('RGB')
            img_t = classifier_transform(img).unsqueeze(0).to(device)
            
            outputs = classifier_model(img_t)
            _, predicted = torch.max(outputs.data, 1)
            
            if predicted.item() == target_label:
                correct += 1
                
    return (correct / total) * 100.0

def main():
    checkpoints = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    
    def sort_key(x):
        if "best" in x: return 999
        try:
            return int(x.split('_')[-1])
        except:
            return 0
    checkpoints.sort(key=sort_key)
    
    print("="*90)
    print(f"{'CHECKPOINT':<32} | {'AGING FID':<10} | {'AGING %':<10} | {'REJUV FID':<10} | {'REJUV %':<10}")
    print("="*90)
    
    results = []
    
    for ckpt in checkpoints:
        aging_dir = os.path.join(RESULTS_DIR, ckpt, 'aging')
        rejuv_dir = os.path.join(RESULTS_DIR, ckpt, 'rejuv')
        
        # FID
        aging_fid = calculate_fid(TEST_SENESCENT_DIR, aging_dir)
        rejuv_fid = calculate_fid(TEST_YOUNG_DIR, rejuv_dir)
        
        # Classifier Fools (%)
        aging_fool_acc = calculate_classifier_score(aging_dir, target_label=0) # senescent=0
        rejuv_fool_acc = calculate_classifier_score(rejuv_dir, target_label=1) # young=1
        
        print(f"{ckpt:<32} | {aging_fid:<10.2f} | %{aging_fool_acc:<9.2f} | {rejuv_fid:<10.2f} | %{rejuv_fool_acc:<9.2f}")
        results.append({
            'ckpt': ckpt,
            'aging_fid': aging_fid,
            'rejuv_fid': rejuv_fid,
            'aging_fool_acc': aging_fool_acc,
            'rejuv_fool_acc': rejuv_fool_acc
        })
        
    print("="*90)
    
    best_aging = min([r for r in results if r['aging_fid'] > 0], key=lambda x: x['aging_fid'])
    best_rejuv = min([r for r in results if r['rejuv_fid'] > 0], key=lambda x: x['rejuv_fid'])
    
    best_aging_fool = max(results, key=lambda x: x['aging_fool_acc'])
    best_rejuv_fool = max(results, key=lambda x: x['rejuv_fool_acc'])
    
    print(f"\n🏆 EN İYİ YAŞLANDIRMA (FID'e göre) : {best_aging['ckpt']} (FID: {best_aging['aging_fid']:.2f})")
    print(f"🥇 EN İYİ YAŞLANDIRMA (Classifere Göre) : {best_aging_fool['ckpt']} (%{best_aging_fool['aging_fool_acc']:.1f} Kandırdı)")
    
    print(f"\n🏆 EN İYİ GENÇLEŞTİRME (FID'e göre): {best_rejuv['ckpt']} (FID: {best_rejuv['rejuv_fid']:.2f})")
    print(f"🥇 EN İYİ GENÇLEŞTİRME (Classifere Göre): {best_rejuv_fool['ckpt']} (%{best_rejuv_fool['rejuv_fool_acc']:.1f} Kandırdı)")

if __name__ == "__main__":
    main()
