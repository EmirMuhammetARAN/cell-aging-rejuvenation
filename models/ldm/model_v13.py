"""
CellLDM v13: Continuous Biological Age-Conditioned Latent Diffusion Model.
Upgrades class-conditioning to continuous biological senescence clock scores (P1 to P40).
Seamlessly initializes from LDM v12 weights and enables exact passage/ratio control.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from copy import deepcopy


class ContinuousAgeEmbedding(nn.Module):
    """
    Interpolates between Young (w0) and Senescent (w1) anchor embeddings
    with a trainable non-linear residual MLP to learn fine-grained passage dynamics.
    Supports:
      - Continuous float scores s in [0.0, 1.0] (where Passage P = s * 40 + 1)
      - Integer labels (0=young, 1=senescent, 2=unconditional) for backward compatibility
    """
    def __init__(self, orig_embed: nn.Embedding):
        super().__init__()
        self.orig_embed = orig_embed
        embed_dim = orig_embed.embedding_dim # 640
        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        # Initialize MLP to zero so at start e(s) is mathematically identical to anchor interpolation
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, labels):
        # Case 1: Float continuous age score in [0.0, 1.0]
        if labels.is_floating_point():
            w0 = self.orig_embed.weight[0] # Young anchor
            w1 = self.orig_embed.weight[1] # Senescent anchor
            w2 = self.orig_embed.weight[2] # Unconditional anchor

            # Handle negative / unconditional flag (-1.0 or > 1.5)
            s = labels.view(-1, 1)
            is_uncond = (s < 0.0) | (s > 1.5)
            
            # Linear base interpolation
            base = (1.0 - s) * w0 + s * w1
            # Biological residual adjustment
            res = self.mlp(s)
            emb = base + res
            
            # Where unconditional is flagged, use w2
            if is_uncond.any():
                emb = torch.where(is_uncond, w2.unsqueeze(0), emb)
            return emb
        else:
            # Case 2: Discrete integer labels (0, 1, 2)
            return self.orig_embed(labels)


class CellLDM_v13(nn.Module):
    def __init__(self, num_classes=2, cfg_drop_prob=0.15, latent_loss_weight=0.05, vae_path=None):
        super().__init__()

        if vae_path is not None:
            self.vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
        else:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True)

        self.num_classes = num_classes
        self.cfg_drop_prob = cfg_drop_prob
        self.latent_loss_weight = latent_loss_weight

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
            num_class_embeds=num_classes + 1,  # 3: young=0, senescent=1, uncond=2
        )

        # Wrap class_embedding with ContinuousAgeEmbedding
        self.unet.class_embedding = ContinuousAgeEmbedding(self.unet.class_embedding)

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            beta_end=0.012,
            prediction_type="epsilon",
        )

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

    def forward(self, images, continuous_ages=None):
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, 
                                   (latents.shape[0],), device=latents.device).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)

        cond_labels = continuous_ages
        if cond_labels is not None and self.training:
            # CFG drop: randomly drop continuous conditioning to unconditional (-1.0)
            drop_mask = torch.rand(cond_labels.shape[0], device=cond_labels.device) < self.cfg_drop_prob
            cond_labels = cond_labels.clone().float()
            cond_labels[drop_mask] = -1.0 # Signals unconditional w2

        noise_pred = self.unet(noisy_latents, timesteps, class_labels=cond_labels).sample
        mse_loss = F.mse_loss(noise_pred, noise)

        if self.latent_loss_weight > 0 and self.training:
            alpha_prod_t = self.train_scheduler.alphas_cumprod.to(latents.device)[timesteps]
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
            pred_x0 = (noisy_latents - (1 - alpha_prod_t).sqrt() * noise_pred) / (alpha_prod_t.sqrt() + 1e-8)
            latent_loss = F.l1_loss(pred_x0, latents)
            return mse_loss + self.latent_loss_weight * latent_loss

        return mse_loss

    @torch.no_grad()
    def sample_passage(self, passage: float, num_samples=1, device='cuda',
                       use_ema=True, guidance_scale=3.0, num_steps=50):
        """
        Samples new MSCs conditioned directly on biological passage (P1 to P40).
        """
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()

        score = float((passage - 1.0) / 40.0)
        score_tensor = torch.full((num_samples,), score, dtype=torch.float32, device=device)
        uncond_tensor = torch.full((num_samples,), -1.0, dtype=torch.float32, device=device)

        latents = torch.randn(num_samples, 4, 64, 64, device=device)
        self.inference_scheduler.set_timesteps(num_steps)

        for t in self.inference_scheduler.timesteps:
            t_batch = t.unsqueeze(0).repeat(num_samples).to(device)
            if guidance_scale > 1.0:
                noise_cond = unet(latents, t_batch, class_labels=score_tensor).sample
                noise_uncond = unet(latents, t_batch, class_labels=uncond_tensor).sample
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = unet(latents, t_batch, class_labels=score_tensor).sample
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        images = self.decode(latents)
        return (images.clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def translate_continuous(self, images, target_passage=None, delta_passage=None,
                             current_passage=None, strength=0.65, num_steps=50,
                             use_ema=True, guidance_scale=3.5, noise=None):
        """
        Continuous trajectory translation for MSC senescence.
        Specify target_passage (e.g. 25.0) OR delta_passage (e.g. +10.0 or -6.0).
        """
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()

        if target_passage is None and delta_passage is not None:
            base_p = current_passage if current_passage is not None else 12.0
            target_passage = base_p + delta_passage

        target_passage = max(1.0, min(41.0, target_passage))
        score = float((target_passage - 1.0) / 40.0)

        num_imgs = images.shape[0]
        score_tensor = torch.full((num_imgs,), score, dtype=torch.float32, device=images.device)
        uncond_tensor = torch.full((num_imgs,), -1.0, dtype=torch.float32, device=images.device)

        latents = self.encode(images)
        self.inference_scheduler.set_timesteps(num_steps)
        timesteps = self.inference_scheduler.timesteps

        start_step = int(len(timesteps) * (1 - strength))
        t_start = timesteps[start_step]

        if noise is None:
            noise = torch.randn_like(latents)
        noisy_latents = self.inference_scheduler.add_noise(latents, noise, t_start)

        for t in timesteps[start_step:]:
            t_batch = t.unsqueeze(0).repeat(num_imgs).to(images.device)
            if guidance_scale > 1.0:
                noise_cond = unet(noisy_latents, t_batch, class_labels=score_tensor).sample
                noise_uncond = unet(noisy_latents, t_batch, class_labels=uncond_tensor).sample
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = unet(noisy_latents, t_batch, class_labels=score_tensor).sample

            noisy_latents = self.inference_scheduler.step(noise_pred, t, noisy_latents).prev_sample

        result = self.decode(noisy_latents)
        return (result.clamp(-1, 1) + 1) / 2
