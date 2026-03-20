import os
import json
import sys

# Ensure offline mode is disabled to allow model downloads
os.environ['HF_HUB_OFFLINE'] = '0'
os.environ['HF_DATASETS_OFFLINE'] = '0'

import torch
import torch.nn as nn
# Workaround for tokenizers version mismatch
try:
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
except ImportError as e:
    import warnings
    warnings.filterwarnings("ignore")
    # Try importing anyway despite version warnings
    import sys
    from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from copy import deepcopy
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig
from lpips import LPIPS


class CellLDM(nn.Module):
    def __init__(self, num_classes=2, cfg_drop_prob=0.15, lpips_weight: float = 0.0):
        super().__init__()
        
        import os
        import json
        # Robust path resolution
        current_file = os.path.abspath(__file__)
        model_dir = os.path.dirname(current_file)  # models/ldm
        models_dir = os.path.dirname(model_dir)  # models
        root_dir = os.path.dirname(models_dir)  # project root
        
        vae_path = os.path.join(root_dir, 'checkpoints', 'vae_finetuned', 'best')
        
        # Load VAE config manually to avoid path validation issues
        config_path = os.path.join(vae_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.vae = AutoencoderKL(**config)
        
        # Load weights from safetensors
        from safetensors.torch import load_file
        weights_path = os.path.join(vae_path, 'diffusion_pytorch_model.safetensors')
        state_dict = load_file(weights_path)
        self.vae.load_state_dict(state_dict)
        
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.num_classes = num_classes
        self.cfg_drop_prob = cfg_drop_prob
        
        # Load UNet (will download on first run and cache)
        self.unet = UNet2DConditionModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="unet",
        )
        
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
        )

        self.class_embed = nn.Embedding(num_classes + 1, 768)
        
        self.loraconfig = LoraConfig(
            r=32,
            lora_alpha=32,
            init_lora_weights="gaussian",
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        self.unet.enable_gradient_checkpointing()
        self.unet = get_peft_model(self.unet, self.loraconfig)

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
            clip_sample=False,
        )

        self.scaling_factor = 0.18215
        self.ema_unet = None
        self.ema_decay = 0.999
        self.lpips_weight = lpips_weight

        self.lpips_loss = None
        if self.lpips_weight > 0.0:
            # LPIPS ağı yalnızca loss hesaplamak için kullanılıyor, eğitilmiyor
            self.lpips_loss = LPIPS(net="vgg")
            self.lpips_loss.eval()
            self.lpips_loss.requires_grad_(False)

    def init_ema(self):
        self.ema_unet = deepcopy(self.unet)
        self.ema_unet.requires_grad_(False)
        self.ema_unet.eval()

    
    @torch.no_grad()
    def update_ema(self):
        if self.ema_unet is None:
            return
        for ema_param, param in zip(self.ema_unet.parameters(), self.unet.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1.0 - self.ema_decay)
    
    @torch.no_grad()
    def encode(self, x):
        latent_dist = self.vae.encode(x).latent_dist
        latents = latent_dist.sample() * self.scaling_factor
        return latents
    
    @torch.no_grad()
    def decode(self, latents):
        images = []
        for i in range(latents.shape[0]):
            torch.cuda.empty_cache()
            img = self.vae.decode(latents[i:i+1] / self.scaling_factor).sample
            images.append(img)
        images = torch.cat(images, dim=0)
        images = (images.clamp(-1, 1) + 1) / 2
        return images

    def forward(self, images, labels=None):
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, 
                                   (latents.shape[0],), device=latents.device).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)
        
        if labels is not None and self.training:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.cfg_drop_prob
            labels = labels.clone()
            labels[drop_mask] = self.num_classes

        class_emb = self.class_embed(labels).unsqueeze(1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=class_emb).sample
        mse_loss = F.mse_loss(noise_pred, noise)

        # Varsayılan: sadece diffusion MSE loss
        if self.lpips_weight <= 0.0 or self.lpips_loss is None or not self.training:
            return mse_loss

        # LPIPS için, predicted original sample'ı manuel olarak hesapla
        # Formül: x0 = (xt - sqrt(1 - alpha_prod_t) * noise_pred) / sqrt(alpha_prod_t)
        alphas_cumprod = self.train_scheduler.alphas_cumprod[timesteps]
        alphas_cumprod = alphas_cumprod.reshape(-1, 1, 1, 1)  # reshape for broadcasting
        
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
        
        pred_x0_latents = (noisy_latents - sqrt_one_minus_alpha_prod * noise_pred) / sqrt_alpha_prod

        # Bu görüntüler VAE'den geçtiği için [0, 1] aralığında
        recon_images = self.decode(pred_x0_latents)

        # LPIPS -1..1 aralığında bekliyor, bu yüzden yeniden ölçekle
        recon_lpips = recon_images * 2.0 - 1.0
        target_lpips = images.clamp(-1, 1)

        # LPIPS modülünü doğru cihaza taşı
        if self.lpips_loss is not None:
            self.lpips_loss.to(images.device)

        lpips_val = self.lpips_loss(recon_lpips, target_lpips).mean()
        total_loss = mse_loss + self.lpips_weight * lpips_val
        return total_loss

    @torch.no_grad()
    def sample(self, num_samples=4, device='cuda', labels=None, use_ema=True, 
               guidance_scale=3.0, num_steps=50):
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()
        
        latents = torch.randn(num_samples, 4, 64, 64, device=device)
        
        self.inference_scheduler.set_timesteps(num_steps)
        
        uncond_labels = torch.full((num_samples,), self.num_classes, dtype=torch.long, device=device)
        
        
        for t in self.inference_scheduler.timesteps:
                t_batch = t.unsqueeze(0).repeat(num_samples).to(device)

                if guidance_scale > 1.0 and labels is not None:
                    class_emb = self.class_embed(labels).unsqueeze(1)
                    noise_pred_cond = unet(latents, t_batch, encoder_hidden_states=class_emb).sample

                    uncond_class_emb = self.class_embed(uncond_labels).unsqueeze(1)
                    noise_pred_uncond = unet(latents, t_batch, encoder_hidden_states=uncond_class_emb).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        images = self.decode(latents)
        return images

    @torch.no_grad()
    def translate(self, images, target_labels, strength=0.55, num_steps=50,
                  use_ema=True, guidance_scale=3.0):
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()
        
        latents = self.encode(images)
        
        self.inference_scheduler.set_timesteps(num_steps)
        timesteps = self.inference_scheduler.timesteps
        
        start_step = int(len(timesteps) * (1 - strength))
        t_start = timesteps[start_step]
        
        noise = torch.randn_like(latents)
        noisy_latents = self.inference_scheduler.add_noise(latents, noise, t_start)
        
        uncond_labels = torch.full((images.shape[0],), self.num_classes, dtype=torch.long, device=images.device)
        
    
        for t in timesteps[start_step:]:
                t_batch = t.unsqueeze(0).repeat(images.shape[0]).to(images.device)

                if guidance_scale > 1.0:
                    target_class_emb = self.class_embed(target_labels).unsqueeze(1)
                    noise_pred_cond = unet(noisy_latents, t_batch, encoder_hidden_states=target_class_emb).sample

                    uncond_class_emb = self.class_embed(uncond_labels).unsqueeze(1)
                    noise_pred_uncond = unet(noisy_latents, t_batch, encoder_hidden_states=uncond_class_emb).sample
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    target_class_emb = self.class_embed(target_labels).unsqueeze(1)
                    noise_pred = unet(noisy_latents, t_batch, encoder_hidden_states=target_class_emb).sample

                noisy_latents = self.inference_scheduler.step(noise_pred, t, noisy_latents).prev_sample
        
        result = self.decode(noisy_latents)
        return result