import os
import sys
from torchvision import transforms
import torch
from PIL import Image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier

# Modeli yükleme
model = Classifier(output_size=2)
model.load_state_dict(torch.load(os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')))
model.to('cuda')
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Grid Search klasörünün yolu
grid_dir = os.path.join(root_dir, 'results', 'generated', 'grid_search_ldm_v2')

print("=" * 70)
print("GRID SEARCH CLASSIFIER ACCURACY TESTİ (MİNİ-TARAMA)")
print("=" * 70)

if not os.path.exists(grid_dir):
    print(f"HATA: {grid_dir} bulunamadı! Önce grid_search üretimini tamamla aga.")
    sys.exit()

# Klasörleri alfabetik sırala ki çıktılar düzenli alt alta gelsin
folders = sorted(os.listdir(grid_dir))

for folder_name in folders:
    folder_path = os.path.join(grid_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    # Klasör ismine göre "Beklenen Sınıfı" otomatik belirliyoruz
    if folder_name.startswith('aging'):
        expected_class = 0  # Yaşlı bekleniyor
    elif folder_name.startswith('rejuv'):
        expected_class = 1  # Genç bekleniyor
    elif folder_name.startswith('random_young'):
        expected_class = 1  # Genç bekleniyor
    elif folder_name.startswith('random_senes'):
        expected_class = 0  # Yaşlı bekleniyor
    else:
        continue # Tanımsız klasörleri atla
    
    correct = 0
    total = 0
    
    for img_name in os.listdir(folder_path):
        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(folder_path, img_name)
        with torch.no_grad():
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0).to('cuda')
            prediction = model(img)
            
            total += 1
            if prediction.argmax(dim=1).item() == expected_class:
                correct += 1
    
    if total > 0:
        acc = correct / total
        # Çıktıyı hizalı (ljust) basıyoruz ki ekranda tablo gibi dursun
        print(f'{folder_name.ljust(40)}: {correct}/{total} = {acc*100:.2f}%')

del model
torch.cuda.empty_cache()

print("-" * 70)
print("Aga Notu: Sadece 10'ar resim olduğu için FID kısmı iptal edildi.")
print("FID hesaplaması sadece final üretimde (300+ resim) yapılmalıdır.")
print("=" * 70)