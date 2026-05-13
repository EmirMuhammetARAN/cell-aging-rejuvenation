"""
GradCAM Görselleştirme - Sınıflandırıcının hücrelerde neye baktığını gösterir.
Çıktılar: gradcam/ klasörüne kaydedilir.
"""
import os, sys, torch, io
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook'ları kaydet
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        self.model.zero_grad()
        target = output[0, target_class]
        target.backward()
        
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, target_class, output


def overlay_heatmap(image_np, cam, alpha=0.5):
    """Isı haritasını orijinal görüntünün üzerine bindir"""
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Turbo colormap (kırmızı=sıcak, mavi=soğuk)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = np.float32(heatmap) * alpha + np.float32(image_np) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return overlay, cam_resized


def create_comparison(original, overlay, cam, pred_label, true_label, confidence):
    """Orijinal + Isı Haritası + Saf CAM yan yana"""
    h, w = original.shape[:2]
    
    # Saf CAM görselleştirme
    cam_colored = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam, (w, h))), cv2.COLORMAP_TURBO)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    
    # Yan yana birleştir
    comparison = np.hstack([original, overlay, cam_colored])
    
    return comparison


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CLASS_NAMES = ['Senescent', 'Young']
    
    # Klasörleri oluştur
    output_dir = os.path.join(root_dir, 'results', 'gradcam')
    os.makedirs(os.path.join(output_dir, 'young'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'senescent'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'grid'), exist_ok=True)
    
    # Sınıflandırıcıyı yükle
    classifier = Classifier(output_size=2)
    ckpt_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    classifier.to(DEVICE)
    classifier.eval()
    
    # ResNet18'in son konvolüsyon katmanı = layer4
    target_layer = classifier.model.layer4[-1]
    gradcam = GradCAM(classifier, target_layer)
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test verilerini yükle
    test_young = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    test_senes = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
    
    all_results = []
    
    for label, (folder, true_class) in [('young', (test_young, 1)), ('senescent', (test_senes, 0))]:
        images = sorted(os.listdir(folder))[:25]  # Her sınıftan 25 örnek
        print(f"\n[{label.upper()}] {len(images)} images processing...")
        
        for img_name in images:
            img_path = os.path.join(folder, img_name)
            
            # Orijinal görüntü (görselleştirme için)
            orig_img = Image.open(img_path).convert('RGB')
            orig_np = np.array(orig_img)
            
            # Model girdisi
            input_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)
            
            # GradCAM
            cam, pred_class, logits = gradcam.generate(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence = probs[0, pred_class].item()
            
            # Isı haritası overlay
            overlay, cam_resized = overlay_heatmap(orig_np, cam)
            
            # Karşılaştırma görseli
            comparison = create_comparison(orig_np, overlay, cam, pred_class, true_class, confidence)
            
            # Kaydet
            base_name = os.path.splitext(img_name)[0]
            pred_label = CLASS_NAMES[pred_class]
            correct = "✓" if pred_class == true_class else "✗"
            
            out_path = os.path.join(output_dir, label, f'{base_name}_gradcam.png')
            Image.fromarray(comparison).save(out_path)
            
            all_results.append({
                'name': img_name, 'true': label, 'pred': pred_class,
                'conf': confidence, 'correct': pred_class == true_class
            })
            
            print(f"  {correct} {img_name}: {pred_label} ({confidence:.1%})")
    
    # Grid oluştur (en iyi 4 young + 4 senescent)
    print("\n--- Creating Grid Images ---")
    
    for label_name, true_cls in [('young', 0), ('senescent', 1)]:
        folder = test_young if true_cls == 0 else test_senes
        images = sorted(os.listdir(folder))[:4]
        
        grid_rows = []
        for img_name in images:
            orig_img = Image.open(os.path.join(folder, img_name)).convert('RGB')
            orig_np = np.array(orig_img)
            input_tensor = transform(orig_img).unsqueeze(0).to(DEVICE)
            cam, pred_class, logits = gradcam.generate(input_tensor)
            overlay, _ = overlay_heatmap(orig_np, cam, alpha=0.45)
            
            # Resize for grid
            h_target = 256
            w_target = 256
            orig_resized = cv2.resize(orig_np, (w_target, h_target))
            overlay_resized = cv2.resize(overlay, (w_target, h_target))
            grid_rows.append(np.hstack([orig_resized, overlay_resized]))
        
        grid = np.vstack(grid_rows)
        grid_path = os.path.join(output_dir, 'grid', f'gradcam_grid_{label_name}.png')
        Image.fromarray(grid).save(grid_path)
        print(f"  OK Grid saved: {grid_path}")
    
    # İstatistik
    correct_count = sum(1 for r in all_results if r['correct'])
    total_count = len(all_results)
    avg_conf = np.mean([r['conf'] for r in all_results])
    
    print(f"\n{'='*60}")
    print(f"GradCAM Analysis Complete!")
    print(f"Accuracy: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)")
    print(f"Avg Confidence: {avg_conf:.1%}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
