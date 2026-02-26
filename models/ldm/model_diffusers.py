import torch
import torch.nn as nn
from diffusers import Unet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from copy import deepcopy
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig

class CellLDM(nn.Module):
    def __init__(self, num_classes=2, cfg_drop_prob=0.15):
        super().__init__()
        
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        )
        
        self.num_classes = num_classes
        self.cfg_drop_prob = cfg_drop_prob
        
        self.unet = Unet2DConditionModel.from_pretrained(
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
        self.class_embed = nn.Sequential(
            nn.Embedding(num_classes + 1, 768)
        )
        
        self.loraconfig = LoraConfig(
            r=64,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_v", "to_q", "to_out.0"],
        )


        self.unet = get_peft_model(self.unet, self.loraconfig)

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
            clip_sample=False,
        )
        self.vae.requires_grad_(False)

        self.scaling_factor = 0.18215
        
        self.ema_unet = None
        self.ema_decay = 0.999
    
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
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, 
                                   (latents.shape[0],), device=latents.device).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)
        
        if labels is not None and self.training:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.cfg_drop_prob
            labels = labels.clone()
            labels[drop_mask] = self.num_classes 

        class_emb = self.class_embed(labels).unsqueeze(1)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=class_emb).sample
        loss = F.mse_loss(noise_pred, noise)
        return loss

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
                class_emb = self.class_embed(labels)
                class_emb = class_emb.unsqueeze(1)
                noise_pred_cond = unet(latents, t_batch, encoder_hidden_states=class_emb).sample

                uncond_class_emb = self.class_embed(uncond_labels)
                uncond_class_emb = uncond_class_emb.unsqueeze(1)
                noise_pred_uncond = unet(latents, t_batch, encoder_hidden_states=uncond_class_emb).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                class_emb = self.class_embed(labels)
                class_emb = class_emb.unsqueeze(1)
                noise_pred = unet(latents, t_batch, encoder_hidden_states=class_emb).sample
            
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample
        
        images = self.decode(latents)
        images = (images.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def translate(self, images, target_labels, strength=0.6, num_steps=50, 
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
                target_class_emb = self.class_embed(target_labels)
                target_class_emb = target_class_emb.unsqueeze(1)
                noise_pred_cond = unet(noisy_latents, t_batch, encoder_hidden_states=target_class_emb).sample

                uncond_class_emb = self.class_embed(uncond_labels)
                uncond_class_emb = uncond_class_emb.unsqueeze(1)
                noise_pred_uncond = unet(noisy_latents, t_batch, encoder_hidden_states=uncond_class_emb).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                target_class_emb = self.class_embed(target_labels)
                target_class_emb = target_class_emb.unsqueeze(1)
                noise_pred = unet(noisy_latents, t_batch, encoder_hidden_states=target_class_emb).sample
            
            noisy_latents = self.inference_scheduler.step(noise_pred, t, noisy_latents).prev_sample
        
        result = self.decode(noisy_latents)
        result = (result.clamp(-1, 1) + 1) / 2
        return result