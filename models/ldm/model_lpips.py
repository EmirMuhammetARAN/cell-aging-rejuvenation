import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from copy import deepcopy
import lpips

class CellLDM(nn.Module):
    def __init__(self, num_classes=2, cfg_drop_prob=0.15, lpips_weight=0.01, vae_path=None):
        super().__init__()

        if vae_path is not None:
            self.vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
        else:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True)

        self.num_classes = num_classes
        self.cfg_drop_prob = cfg_drop_prob

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
            num_class_embeds=num_classes + 1,  
        )

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
        # Pretrained SD VAE: latent std ~5.5, scale by 0.18215 to normalize
        # Custom fine-tuned VAE: latent std ~1.0, no scaling needed
        self.scaling_factor = 1.0 if vae_path is not None else 0.18215

        self.ema_unet = None
        self.ema_decay = 0.999

        # Latent-space structural loss weight (replaces pixel LPIPS constraints)
        self.lpips_weight = lpips_weight

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

        noise_pred = self.unet(noisy_latents, timesteps, class_labels=labels).sample
        mse_loss = F.mse_loss(noise_pred, noise)

        # Latent-space structural loss: predict x0 and compare directly to true latents
        # This completely avoids the VAE decoder's massive overhead while enforcing structural accuracy
        if self.lpips_weight > 0 and self.training:
            alpha_prod_t = self.train_scheduler.alphas_cumprod.to(latents.device)[timesteps]
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
            pred_x0 = (noisy_latents - (1 - alpha_prod_t).sqrt() * noise_pred) / (alpha_prod_t.sqrt() + 1e-8)
            
            latent_loss = F.l1_loss(pred_x0, latents)
            return mse_loss + self.lpips_weight * latent_loss
        
        return mse_loss

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
                noise_pred_cond = unet(latents, t_batch, class_labels=labels).sample
                noise_pred_uncond = unet(latents, t_batch, class_labels=uncond_labels).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = unet(latents, t_batch, class_labels=labels).sample
            
            latents = self.inference_scheduler.step(noise_pred, t, latents).prev_sample

        images = self.decode(latents)
        images = (images.clamp(-1, 1) + 1) / 2
        return images

    @torch.no_grad()
    def translate(self, images, target_labels, strength=0.6, num_steps=50, 
                  use_ema=True, guidance_scale=3.0, noise=None):
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()

        latents = self.encode(images)

        self.inference_scheduler.set_timesteps(num_steps)
        timesteps = self.inference_scheduler.timesteps

        start_step = int(len(timesteps) * (1 - strength))
        t_start = timesteps[start_step]

        if noise is None:
            noise = torch.randn_like(latents)
        noisy_latents = self.inference_scheduler.add_noise(latents, noise, t_start)
        
        uncond_labels = torch.full((images.shape[0],), self.num_classes, dtype=torch.long, device=images.device)

        for t in timesteps[start_step:]:
            t_batch = t.unsqueeze(0).repeat(images.shape[0]).to(images.device)
            
            if guidance_scale > 1.0:
                noise_pred_cond = unet(noisy_latents, t_batch, class_labels=target_labels).sample
                noise_pred_uncond = unet(noisy_latents, t_batch, class_labels=uncond_labels).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = unet(noisy_latents, t_batch, class_labels=target_labels).sample
            
            noisy_latents = self.inference_scheduler.step(noise_pred, t, noisy_latents).prev_sample

        result = self.decode(noisy_latents)
        result = (result.clamp(-1, 1) + 1) / 2
        return result

    @torch.no_grad()
    def translate_masked(self, images, target_labels, cell_mask,
                         strength=0.6, num_steps=50,
                         use_ema=True, guidance_scale=3.0,
                         dilation_px=8, noise=None):
        """
        Mask-guided SDEdit for MSC senescence translation.

        Adapts the RePaint inpainting strategy (Lugmayr et al., CVPR 2022)
        to SDEdit-style image-to-image translation for MSC morphology.
        At each denoising step the background region (outside the dilated
        cell mask) is kept anchored to the correspondingly-noised original
        latent, so only the foreground cell is transformed.

        Args:
            images:        Input image tensor, shape (B, 3, H, W), range [-1,1].
            target_labels: Class label tensor, shape (B,).
            cell_mask:     Binary foreground mask, shape (B, 1, H, W) or
                           (B, H, W), values in {0, 1}, pixel space (H x W).
                           Typically the output of Mask R-CNN thresholded to
                           a single foreground region.
            strength:      SDEdit noise strength (0=no change, 1=full denoise).
            num_steps:     Number of DDIM inference steps.
            use_ema:       Whether to use EMA weights.
            guidance_scale: Classifier-free guidance scale.
            dilation_px:   Morphological dilation radius (pixels) applied to
                           the mask before downsampling to latent space, to
                           create a soft editing boundary and reduce artefacts
                           at the cell boundary.
            noise:         Optional pre-drawn noise tensor (B, 4, 64, 64) for
                           strictly paired comparison with unmasked SDEdit.

        Returns:
            Translated image tensor, shape (B, 3, H, W), range [0, 1].
        """
        unet = self.ema_unet if (use_ema and self.ema_unet is not None) else self.unet
        unet.eval()

        # --- 1. Encode original image to latent space ---
        latents_orig = self.encode(images)          # (B, 4, 64, 64)

        # --- 2. Build latent-space mask ---
        if cell_mask.dim() == 3:
            cell_mask = cell_mask.unsqueeze(1)      # (B, 1, H, W)
        mask_px = cell_mask.float().to(images.device)

        # Dilate in pixel space with max-pool to soften the boundary
        if dilation_px > 0:
            mask_px = F.max_pool2d(
                mask_px,
                kernel_size=2 * dilation_px + 1,
                stride=1,
                padding=dilation_px,
            )

        # Downsample to latent resolution (512 -> 64, factor 8)
        lat_h, lat_w = latents_orig.shape[2], latents_orig.shape[3]
        mask_lat = F.interpolate(
            mask_px, size=(lat_h, lat_w), mode='nearest'
        )                                           # (B, 1, 64, 64)
        mask_lat = mask_lat.expand_as(latents_orig) # (B, 4, 64, 64)

        # --- 3. Add noise up to the start timestep (same as translate) ---
        self.inference_scheduler.set_timesteps(num_steps)
        timesteps = self.inference_scheduler.timesteps

        start_step = int(len(timesteps) * (1 - strength))
        t_start = timesteps[start_step]

        if noise is None:
            noise = torch.randn_like(latents_orig)
        noisy_latents = self.inference_scheduler.add_noise(latents_orig, noise, t_start)

        uncond_labels = torch.full(
            (images.shape[0],), self.num_classes,
            dtype=torch.long, device=images.device
        )

        # --- 4. Denoising loop with background anchoring ---
        step_timesteps = timesteps[start_step:]
        for i, t in enumerate(step_timesteps):
            t_batch = t.unsqueeze(0).repeat(images.shape[0]).to(images.device)

            # Classifier-free guidance
            if guidance_scale > 1.0:
                noise_pred_cond   = unet(noisy_latents, t_batch, class_labels=target_labels).sample
                noise_pred_uncond = unet(noisy_latents, t_batch, class_labels=uncond_labels).sample
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = unet(noisy_latents, t_batch, class_labels=target_labels).sample

            noisy_latents = self.inference_scheduler.step(noise_pred, t, noisy_latents).prev_sample

            # --- RePaint-style background anchoring (correctly aligned to t_prev) ---
            # scheduler.step() transitions the sample to the next timestep in the schedule.
            # Background must be anchored to the original latent at t_prev, not the previous t.
            if i < len(step_timesteps) - 1:
                t_prev = step_timesteps[i + 1].reshape(1).to(images.device)
                noisy_orig_prev = self.inference_scheduler.add_noise(latents_orig, noise, t_prev)
            else:
                # Final step: sample has fully denoised to t=0
                noisy_orig_prev = latents_orig

            # Blend: foreground from denoised prediction, background from anchored original
            noisy_latents = mask_lat * noisy_latents + (1.0 - mask_lat) * noisy_orig_prev

        result = self.decode(noisy_latents)
        result = (result.clamp(-1, 1) + 1) / 2
        return result