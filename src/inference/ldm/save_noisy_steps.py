import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

current_file = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.insert(0, root_dir)

from models.ldm.model_lpips import CellLDM

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Checkpoint
    ckpt_path = os.path.join(root_dir, 'checkpoints', 'ldm', 'checkpoint_v12_v4_data_lpips_last.pt')
    
    # Image
    img_path = os.path.join(root_dir, 'data', 'processed_v4', 'test', 'senescent', 'senescent_MSCs_10005_2.jpg')
    
    # Output directory
    out_dir = os.path.join(root_dir, 'results', 'generated', 'noise_visualization')
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = CellLDM(num_classes=2)
    model.to(device, memory_format=torch.channels_last)
    model.vae.to(memory_format=torch.channels_last)
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    model.unet.load_state_dict(checkpoint['unet_state_dict'])
    model.init_ema()
    if 'ema_unet_state_dict' in checkpoint:
        model.ema_unet.load_state_dict(checkpoint['ema_unet_state_dict'])
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Prepare image
    img = Image.open(img_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device, memory_format=torch.channels_last)
    
    # Original resim (Kontrol için)
    save_image((x.cpu() * 0.5 + 0.5), os.path.join(out_dir, 'step_0_original.png'))
    
    with torch.no_grad():
        # Encode to latent
        posterior = model.vae.encode(x).latent_dist
        latents = posterior.sample() * model.scaling_factor
        
        # Setup scheduler
        num_steps = 100
        model.inference_scheduler.set_timesteps(num_steps, device=device)
        
        # Sabit noise (hep aynı gürültü paternini görmek için)
        torch.manual_seed(2026) 
        noise = torch.randn_like(latents)
        
        # Farklı strength seviyeleri (ne kadar bozulduğunu göstermek için makaleye çok yakışır)
        strengths = [0.25, 0.50, 0.65, 0.75, 1.0]
        
        for strength in strengths:
            init_step = int(num_steps * strength)
            t_start = model.inference_scheduler.timesteps[-init_step].unsqueeze(0)
            
            # Gürültü ekle (Forward diffusion)
            noisy_latents = model.inference_scheduler.add_noise(latents, noise, t_start)
            
            # VAE ile geri çöz (Piksel uzayına al ki gözle görelim)
            noisy_latents = noisy_latents / model.scaling_factor
            decoded = model.vae.decode(noisy_latents).sample
            
            # Kaydet
            res = (decoded.cpu() * 0.5 + 0.5).clamp(0, 1)
            out_name = os.path.join(out_dir, f'step_{int(strength*100)}percent_noise.png')
            save_image(res, out_name)
            print(f"Saved: {out_name}")

if __name__ == "__main__":
    main()
