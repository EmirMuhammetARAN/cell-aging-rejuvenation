"""
Preprint / Thesis Figure 4: Explainable AI with Hierarchical LayerCAM (Layer 3 + Layer 4)
Uses Classifier v2 (ResNet-18, classifier_v2.pth) to visualize subcellular phenotypic attention
across Young and Senescent MSC single-cell crops.
Outputs:
  - results/thesis_figures/preprint_figure4_gradcam.png
  - results/gradcam/grid/gradcam_grid_young.png
  - results/gradcam/grid/gradcam_grid_senescent.png
"""
import os
import sys
import shutil
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models.classifier.classifier import Classifier

ARTIFACT_DIR = r'C:\Users\emir_\.gemini\antigravity-ide\brain\5ba6371e-e7f8-4738-bb94-e1efd97424b7'


class HierarchicalLayerCAM:
    """
    Hierarchical LayerCAM implementation for ResNet architectures.
    Extracts element-wise positive gradient weighted activations from Layer 3 and Layer 4,
    combining fine spatial resolution (32x32) with deep semantic certainty (16x16).
    """
    def __init__(self, model):
        self.model = model
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def get_hook(name):
            def forward_hook(module, input, output):
                self.activations[name] = output.detach()
            def backward_hook(module, grad_input, grad_output):
                self.gradients[name] = grad_output[0].detach()
            return forward_hook, backward_hook

        for name, module in [('layer3', self.model.model.layer3), ('layer4', self.model.model.layer4)]:
            f_hook, b_hook = get_hook(name)
            self.hooks.append(module.register_forward_hook(f_hook))
            self.hooks.append(module.register_full_backward_hook(b_hook))

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(input_tensor)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        score = logits[0, target_class]
        score.backward(retain_graph=True)
        
        cam_maps = []
        for name in ['layer3', 'layer4']:
            acts = self.activations[name][0]     # [C, H, W]
            grads = self.gradients[name][0]     # [C, H, W]
            
            # Element-wise positive gradient weighting (LayerCAM)
            weights = F.relu(grads)
            weighted_acts = weights * acts
            cam = torch.sum(weighted_acts, dim=0) # [H, W]
            cam = F.relu(cam)
            
            cam_np = cam.cpu().numpy()
            c_min, c_max = np.min(cam_np), np.max(cam_np)
            if c_max > c_min:
                cam_np = (cam_np - c_min) / (c_max - c_min)
            else:
                cam_np = np.zeros_like(cam_np)
                
            # Upsample to 512x512
            cam_resized = cv2.resize(cam_np, (512, 512), interpolation=cv2.INTER_LINEAR)
            cam_maps.append(cam_resized)
            
        # Hierarchical Fusion: 50% Layer 3 (High Res) + 50% Layer 4 (High Semantics)
        fused_cam = 0.5 * cam_maps[0] + 0.5 * cam_maps[1]
        f_min, f_max = np.min(fused_cam), np.max(fused_cam)
        if f_max > f_min:
            fused_cam = (fused_cam - f_min) / (f_max - f_min)
            
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        return fused_cam, probs

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


def overlay_heatmap(image_np, cam, alpha=0.48):
    """Overlays LayerCAM attention heatmap onto raw cell crop."""
    h, w = image_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # TURBO colormap for optimal scientific dynamic range
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.float32(heatmap) * alpha + np.float32(image_np) * (1.0 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


def get_font(size=14, bold=False):
    font_names = ["arialbd.ttf" if bold else "arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except Exception:
            continue
    return ImageFont.load_default()


def main():
    print("[*] Initializing Classifier v2 and Hierarchical LayerCAM (Layer 3 + Layer 4)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    classifier = Classifier(output_size=2).to(device)
    ckpt_path = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
    classifier.load_state_dict(torch.load(ckpt_path, map_location=device))
    classifier.eval()
    print("    [OK] Loaded checkpoint: checkpoints/classifier/classifier_v2.pth")
    
    layercam = HierarchicalLayerCAM(classifier)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_young = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'young')
    test_senes = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent')
    
    grid_out_dir = os.path.join(root_dir, 'results', 'gradcam', 'grid')
    thesis_dir = os.path.join(root_dir, 'results', 'thesis_figures')
    os.makedirs(grid_out_dir, exist_ok=True)
    os.makedirs(thesis_dir, exist_ok=True)
    
    # 4 high-confidence young and 4 high-confidence senescent cells
    selected_young = [
        'senescent_MSCs_10005_16.jpg',  # Young 86.9%
        'senescent_MSCs_10005_5.jpg',   # Young 88.2%
        'senescent_MSCs_10010_39.jpg',  # Young 91.9%
        'senescent_MSCs_10030_47.jpg',  # Young 93.8%
    ]
    selected_senes = [
        'senescent_MSCs_10005_0.jpg',   # Senes 94.6%
        'senescent_MSCs_10005_1.jpg',   # Senes 99.9%
        'senescent_MSCs_10005_10.jpg',  # Senes 99.8%
        'senescent_MSCs_10005_12.jpg',  # Senes 99.9%
    ]
    
    cell_size = 256
    sub_header_h = 24
    panel_header_h = 36
    
    font_panel = get_font(15, bold=True)
    font_sub = get_font(12, bold=True)
    font_badge = get_font(11, bold=True)
    
    panels = {}
    
    for label_name, files, folder, target_cls, header_color in [
        ('young', selected_young, test_young, 1, (44, 160, 44)),
        ('senescent', selected_senes, test_senes, 0, (214, 39, 40))
    ]:
        print(f"[*] Processing {label_name.upper()} panel (4 samples)...")
        w_panel = cell_size * 2
        h_panel = panel_header_h + sub_header_h + cell_size * 4
        panel_img = Image.new('RGB', (w_panel, h_panel), 'white')
        draw_p = ImageDraw.Draw(panel_img)
        
        # Main panel header
        panel_title = f"{label_name.capitalize()} Cells (Classifier v2)"
        draw_p.rectangle([0, 0, w_panel, panel_header_h], fill=header_color)
        bbox = draw_p.textbbox((0, 0), panel_title, font=font_panel)
        tw = bbox[2] - bbox[0]
        draw_p.text(((w_panel - tw) // 2, 8), panel_title, fill='white', font=font_panel)
        
        # Subheaders: Original | LayerCAM (L3+L4)
        draw_p.rectangle([0, panel_header_h, cell_size, panel_header_h + sub_header_h], fill='#f0f0f0')
        draw_p.rectangle([cell_size, panel_header_h, w_panel, panel_header_h + sub_header_h], fill='#e8e8e8')
        
        txt_orig = "Raw Microscopic Crop"
        txt_cam = "LayerCAM Attention (L3+L4)"
        
        bbox_o = draw_p.textbbox((0, 0), txt_orig, font=font_sub)
        draw_p.text(((cell_size - (bbox_o[2] - bbox_o[0])) // 2, panel_header_h + 4), txt_orig, fill='#333333', font=font_sub)
        
        bbox_c = draw_p.textbbox((0, 0), txt_cam, font=font_sub)
        draw_p.text((cell_size + (cell_size - (bbox_c[2] - bbox_c[0])) // 2, panel_header_h + 4), txt_cam, fill='#222222', font=font_sub)
        
        y_cursor = panel_header_h + sub_header_h
        
        for idx, fn in enumerate(files):
            orig_pil = Image.open(os.path.join(folder, fn)).convert('RGB')
            orig_np = np.array(orig_pil)
            
            in_t = transform(orig_pil).unsqueeze(0).to(device)
            cam, probs = layercam.generate_cam(in_t, target_class=target_cls)
            conf = probs[target_cls]
            
            overlay_np = overlay_heatmap(orig_np, cam, alpha=0.48)
            
            orig_res = Image.fromarray(orig_np).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            over_res = Image.fromarray(overlay_np).resize((cell_size, cell_size), Image.Resampling.LANCZOS)
            
            # Badge on overlay
            draw_ov = ImageDraw.Draw(over_res)
            badge_text = f"{'Young' if target_cls == 1 else 'Senescent'}: {conf:.1%}"
            bbox_b = draw_ov.textbbox((0, 0), badge_text, font=font_badge)
            bw = bbox_b[2] - bbox_b[0] + 10
            bh = bbox_b[3] - bbox_b[1] + 6
            # Semi-transparent dark pill
            draw_ov.rounded_rectangle([cell_size - bw - 6, cell_size - bh - 6, cell_size - 6, cell_size - 6], radius=4, fill=(0, 0, 0, 180))
            draw_ov.text((cell_size - bw - 1, cell_size - bh - 4), badge_text, fill='white', font=font_badge)
            
            panel_img.paste(orig_res, (0, y_cursor))
            panel_img.paste(over_res, (cell_size, y_cursor))
            
            # Subtle cell separator lines
            draw_p.line([(0, y_cursor), (w_panel, y_cursor)], fill='#dcdcdc', width=1)
            y_cursor += cell_size
            
        panels[label_name] = panel_img
        
        # Save individual panel grid
        grid_file = os.path.join(grid_out_dir, f"gradcam_grid_{label_name}.png")
        panel_img.save(grid_file, quality=95)
        print(f"    [OK] Saved grid: {grid_file}")
    
    # ── COMBINE INTO SINGLE PREPRINT FIGURE ───────────────────────────────
    gap = 14
    w_tot = cell_size * 4 + gap
    h_tot = panel_header_h + sub_header_h + cell_size * 4
    
    final_canvas = Image.new('RGB', (w_tot, h_tot), '#f4f4f4')
    final_canvas.paste(panels['young'], (0, 0))
    final_canvas.paste(panels['senescent'], (cell_size * 2 + gap, 0))
    
    out_fig = os.path.join(thesis_dir, 'preprint_figure4_gradcam.png')
    final_canvas.save(out_fig, quality=95)
    print(f"\n[OK] Saved master preprint figure: {out_fig}")
    
    if os.path.exists(ARTIFACT_DIR):
        art_out = os.path.join(ARTIFACT_DIR, 'preprint_figure4_gradcam.png')
        shutil.copy2(out_fig, art_out)
        print(f"[OK] Copied to artifact directory: {art_out}")
        
    layercam.remove_hooks()
    print("[DONE] Explainable AI LayerCAM figure generation complete!")


if __name__ == '__main__':
    main()

