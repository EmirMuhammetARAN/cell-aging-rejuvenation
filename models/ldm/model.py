import torch
import torch.nn as nn
from diffusers import UNet2DModel, AutoencoderKL, DDPMScheduler
from copy import deepcopy

class CellLDM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        )
        
        self.num_classes = num_classes
        
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(160, 320, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            attention_head_dim=8,
            norm_num_groups=32,
            num_class_embeds=num_classes,
        )
        
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
        )
        
        self.vae.requires_grad_(False)
        self.scaling_factor = 0.18215
        
        self.ema_unet = None
        self.ema_decay = 0.995
    
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
        latents = latents / self.scaling_factor
        image = self.vae.decode(latents).sample
        return image

    def forward(self, images, labels=None):
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, 
                                   (latents.shape[0],), device=latents.device).long()
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        noise_pred = self.unet(noisy_latents, timesteps, class_labels=labels).sample
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, num_samples=4, device='cuda', labels=None, use_ema=True):
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()
        
        latents = torch.randn(num_samples, 4, 64, 64, device=device)
        
        self.scheduler.set_timesteps(100)
        
        for t in self.scheduler.timesteps:
            t_batch = t.unsqueeze(0).repeat(num_samples).to(device)
            noise_pred = unet(latents, t_batch, class_labels=labels).sample
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        images = self.decode(latents)
        images = (images.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def translate(self, images, target_labels, strength=0.6, num_steps=100, use_ema=True):
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()
        
        latents = self.encode(images)
        
        self.scheduler.set_timesteps(num_steps)
        timesteps = self.scheduler.timesteps
        
        start_step = int(len(timesteps) * (1 - strength))
        t_start = timesteps[start_step]
        
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, t_start)
        
        for t in timesteps[start_step:]:
            t_batch = t.unsqueeze(0).repeat(images.shape[0]).to(images.device)
            noise_pred = unet(noisy_latents, t_batch, class_labels=target_labels).sample
            noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents).prev_sample
        
        result = self.decode(noisy_latents)
        result = (result.clamp(-1, 1) + 1) / 2
        return result