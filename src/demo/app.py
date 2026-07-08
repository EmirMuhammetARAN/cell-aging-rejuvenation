"""
Cell Aging & Rejuvenation Demo - Gradio Interface
"""
import os, sys, torch, gc
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import cv2
import gradio as gr

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM
from models.classifier.classifier import Classifier

# ===== CONFIG =====
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
CLASSIFIER_PATH = os.path.join(root_dir, 'checkpoints', 'classifier', 'classifier_v2.pth')
CLASS_NAMES = {0: 'Senescent', 1: 'Young'}

# ===== MODEL LOADING =====
print("Loading LDM model...", flush=True)
ldm = CellLDM(num_classes=2, lpips_weight=0.0)
ldm.scaling_factor = 0.18215
ldm.to(DEVICE, memory_format=torch.channels_last)
ldm.vae.to(memory_format=torch.channels_last)
ldm.init_ema()

ckpt = torch.load(CHECKPOINT, map_location='cpu')
ldm.unet.load_state_dict(ckpt['unet_state_dict'])
if 'ema_unet_state_dict' in ckpt:
    ldm.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'])
print(f"LDM loaded (epoch={ckpt.get('epoch', '?')})", flush=True)
del ckpt; gc.collect()
ldm.eval()

print("Loading Classifier...", flush=True)
classifier = Classifier(output_size=2)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location='cpu'))
classifier.to(DEVICE)
classifier.eval()

# GradCAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)
    
    def _fwd(self, m, i, o): self.activations = o.detach()
    def _bwd(self, m, gi, go): self.gradients = go[0].detach()
    
    def generate(self, x, target_class=None):
        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(1).item()
        self.model.zero_grad()
        out[0, target_class].backward()
        w = self.gradients.mean(dim=[2,3], keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = F.softmax(out, dim=1)
        return cam, target_class, probs[0].detach().cpu().numpy()

gradcam = GradCAM(classifier, classifier.model.layer4[-1])

# Transforms
ldm_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

cls_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def classify_cell(image):
    """Classify a cell image and return GradCAM overlay"""
    if image is None:
        return None, "Image not loaded"
    
    img = Image.fromarray(image).convert('RGB').resize((512, 512))
    img_np = np.array(img)
    
    input_tensor = cls_transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.enable_grad():
        cam, pred_class, probs = gradcam.generate(input_tensor)
    
    # Overlay
    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_TURBO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.clip(np.float32(heatmap) * 0.45 + np.float32(img_np) * 0.55, 0, 255).astype(np.uint8)
    
    label = CLASS_NAMES[pred_class]
    confidence = probs[pred_class] * 100
    
    result_text = f"Prediction: {label}\nConfidence: {confidence:.1f}%\n\nYoung: {probs[1]*100:.1f}%\nSenescent: {probs[0]*100:.1f}%"
    
    return overlay, result_text


def translate_cell(image, direction, strength, cfg_scale, num_steps):
    """Translate cell between young and senescent states"""
    if image is None:
        return None, None, "Image not loaded"
    
    img = Image.fromarray(image).convert('RGB').resize((512, 512))
    img_np = np.array(img)
    
    # LDM translation
    input_tensor = ldm_transform(img).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
    
    if direction == "Aging (Young -> Senescent)":
        target_label = torch.tensor([1], device=DEVICE)  # LDM: 1 = Senescent
    else:
        target_label = torch.tensor([0], device=DEVICE)  # LDM: 0 = Young
    
    with torch.no_grad():
        output = ldm.translate(
            input_tensor, target_label,
            strength=strength, num_steps=int(num_steps),
            use_ema=True, guidance_scale=cfg_scale
        )
    
    # translate() zaten [0,1] araliginda donduruyor
    output_img = output.squeeze().cpu()
    output_np = (output_img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # Classify both
    orig_tensor = cls_transform(img).unsqueeze(0).to(DEVICE)
    out_pil = Image.fromarray(output_np)
    out_tensor = cls_transform(out_pil).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        orig_pred = F.softmax(classifier(orig_tensor), dim=1)[0].cpu().numpy()
        out_pred = F.softmax(classifier(out_tensor), dim=1)[0].cpu().numpy()
    
    orig_class = CLASS_NAMES[orig_pred.argmax()]
    out_class = CLASS_NAMES[out_pred.argmax()]
    
    result_text = (
        f"--- ORIGINAL ---\n"
        f"Class: {orig_class}\n"
        f"Young: {orig_pred[1]*100:.1f}% | Senescent: {orig_pred[0]*100:.1f}%\n\n"
        f"--- TRANSLATED ---\n"
        f"Class: {out_class}\n"
        f"Young: {out_pred[1]*100:.1f}% | Senescent: {out_pred[0]*100:.1f}%\n\n"
        f"--- PARAMETERS ---\n"
        f"Direction: {direction}\n"
        f"Strength: {strength} | CFG: {cfg_scale} | Steps: {int(num_steps)}"
    )
    
    torch.cuda.empty_cache()
    
    return output_np, result_text


def random_generate(cell_type, num_steps):
    """Generate random cell images from scratch"""
    if cell_type == "Young":
        labels = torch.zeros(4, dtype=torch.long, device=DEVICE)   # LDM: 0 = Young
    else:
        labels = torch.ones(4, dtype=torch.long, device=DEVICE)    # LDM: 1 = Senescent
    
    with torch.no_grad():
        samples = ldm.sample(
            num_samples=4, device=DEVICE, labels=labels,
            use_ema=True, guidance_scale=3.0, num_steps=int(num_steps)
        )
    
    # sample() zaten [0,1] araliginda donduruyor
    samples = samples.clamp(0, 1)
    grid = torch.zeros(3, 512*2, 512*2)
    for i in range(4):
        r, c = i // 2, i % 2
        grid[:, r*512:(r+1)*512, c*512:(c+1)*512] = samples[i]
    
    grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    torch.cuda.empty_cache()
    
    return grid_np


# ===== GRADIO INTERFACE =====
with gr.Blocks(title="Cell Aging & Rejuvenation - AI Demo") as demo:
    
    gr.Markdown("""
    # Cell Aging & Rejuvenation AI Demo
    ### Cellular Aging and Rejuvenation Simulation via Latent Diffusion Model
    """)
    
    with gr.Tabs():
        # Tab 1: Translation
        with gr.TabItem("Cell Translation"):
            gr.Markdown("Upload a cell image to age or rejuvenate it using SDEdit.")
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="Input Cell Image", type="numpy")
                    direction = gr.Radio(
                        ["Aging (Young -> Senescent)", "Rejuvenation (Senescent -> Young)"],
                        label="Direction", value="Aging (Young -> Senescent)"
                    )
                    with gr.Row():
                        strength = gr.Slider(0.3, 1.0, value=0.8, step=0.05, label="Denoising Strength")
                        cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG Scale")
                    steps = gr.Slider(20, 100, value=50, step=10, label="Number of Steps")
                    translate_btn = gr.Button("Translate!", variant="primary")
                
                with gr.Column(scale=1):
                    output_img = gr.Image(label="Translated Cell")
                    result_text = gr.Textbox(label="Classification Results", lines=10)
            
            translate_btn.click(
                translate_cell,
                inputs=[input_img, direction, strength, cfg, steps],
                outputs=[output_img, result_text]
            )
        
        # Tab 2: Classification + GradCAM
        with gr.TabItem("Classification + GradCAM"):
            gr.Markdown("Upload a cell image to classify it and see where the model focuses.")
            with gr.Row():
                with gr.Column(scale=1):
                    cls_input = gr.Image(label="Input Cell Image", type="numpy")
                    cls_btn = gr.Button("Classify + GradCAM", variant="primary")
                with gr.Column(scale=1):
                    gradcam_output = gr.Image(label="GradCAM Heatmap")
                    cls_result = gr.Textbox(label="Classification Result", lines=6)
            
            cls_btn.click(
                classify_cell,
                inputs=[cls_input],
                outputs=[gradcam_output, cls_result]
            )
        
        # Tab 3: Random Generation
        with gr.TabItem("Random Generation (From Scratch)"):
            gr.Markdown("Generate random cell images from scratch.")
            with gr.Row():
                with gr.Column(scale=1):
                    gen_type = gr.Radio(["Young", "Senescent"], label="Cell Type", value="Young")
                    gen_steps = gr.Slider(20, 100, value=50, step=10, label="Number of Steps")
                    gen_btn = gr.Button("Generate 4 Cells!", variant="primary")
                with gr.Column(scale=1):
                    gen_output = gr.Image(label="Generated Cells (2x2 Grid)")
            
            gen_btn.click(
                random_generate,
                inputs=[gen_type, gen_steps],
                outputs=[gen_output]
            )
    
    # Example images
    example_dir = os.path.join(root_dir, 'data', 'processed_v4', 'test')
    young_examples = [os.path.join(example_dir, 'young', f) for f in sorted(os.listdir(os.path.join(example_dir, 'young')))[:3]]
    senes_examples = [os.path.join(example_dir, 'senescent', f) for f in sorted(os.listdir(os.path.join(example_dir, 'senescent')))[:3]]


if __name__ == "__main__":
    print(r"\\nStarting demo server...", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
