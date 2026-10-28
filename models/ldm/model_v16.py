"""
CellLDM v16: Passage-Conditioned Latent Diffusion Model (Fibroblast Stage 1 -> MSC Stage 2)
============================================================================================
Differences w.r.t. CellLDM (v12/v14/v15), which use a 2-entry binary class embedding:

  * Conditioning is a CONTINUOUS cumulative-passage value (1..42), not young/senescent.
      embedding = MLP(sinusoidal(passage)) + cell_type_embedding
  * `cell_type` (0 = Fibroblast, 1 = MSC) is a separate learned embedding so that Stage 2 can
    fine-tune on MSC (pseudo-passage labels) without overwriting the fibroblast passage prior.
  * Classifier-free guidance uses a single learned NULL vector that replaces ONLY the passage
    part of the conditioning (the cell_type embedding is kept), so "unconditional on passage"
    is still cell-type aware.
  * UNet is trained from scratch (no SD UNet weights, no MSC exposure) so the Stage-1 prior is
    purely fibroblast and the cross-cell transfer claim stays clean.
  * EMA covers both the UNet and the conditioner, with a warm-up (from-scratch friendly).

The frozen SD VAE (stabilityai/sd-vae-ft-mse, scale 0.18215) is identical to previous versions.
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DModel, AutoencoderKL, DDPMScheduler, DDIMScheduler

CELL_FIBROBLAST = 0
CELL_MSC = 1


def sinusoidal_embedding(x, dim, max_period=1000.0):
    """x: (B,) float tensor -> (B, dim) sin/cos features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=x.device, dtype=torch.float32) / half
    )
    args = x.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class PassageConditioner(nn.Module):
    """Maps (passage, cell_type) -> a (B, embed_dim) vector added to the UNet time embedding."""

    def __init__(self, embed_dim=640, freq_dim=256, num_cell_types=2, max_passage=42.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.freq_dim = freq_dim
        self.max_passage = float(max_passage)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.cell_type_emb = nn.Embedding(num_cell_types, embed_dim)
        nn.init.zeros_(self.cell_type_emb.weight)
        self.null_vec = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, passage, cell_type, drop_mask=None):
        # normalized age in ~[0, 1]; mild clamp allows tiny extrapolation but avoids garbage
        a = ((passage.float() - 1.0) / (self.max_passage - 1.0)).clamp(-0.05, 1.10)
        e = self.mlp(sinusoidal_embedding(a * 100.0, self.freq_dim))
        if drop_mask is not None:
            e = torch.where(drop_mask[:, None], self.null_vec[None, :].expand_as(e), e)
        return e + self.cell_type_emb(cell_type)


class CellLDM_v16(nn.Module):
    def __init__(self, cfg_drop_prob=0.10, lpips_weight=0.1, max_passage=42.0,
                 ema_decay=0.999, vae_path=None):
        super().__init__()
        if vae_path is not None:
            self.vae = AutoencoderKL.from_pretrained(vae_path, local_files_only=True)
        else:
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", local_files_only=True)
        self.vae.requires_grad_(False)
        self.scaling_factor = 1.0 if vae_path is not None else 0.18215

        self.cfg_drop_prob = cfg_drop_prob
        self.lpips_weight = lpips_weight  # latent-space x0 L1 weight (name kept for consistency)
        self.max_passage = max_passage

        block_out = (160, 320, 512, 512)
        time_embed_dim = block_out[0] * 4  # 640, diffusers convention
        self.unet = UNet2DModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=block_out,
            down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
            up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
            attention_head_dim=8,
            norm_num_groups=32,
            class_embed_type="identity",
        )
        self.cond = PassageConditioner(embed_dim=time_embed_dim, max_passage=max_passage)

        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=1000, beta_schedule="scaled_linear",
            beta_start=0.00085, beta_end=0.012, prediction_type="epsilon",
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=1000, beta_schedule="scaled_linear",
            beta_start=0.00085, beta_end=0.012, prediction_type="epsilon", clip_sample=False,
        )

        self.ema_unet = None
        self.ema_cond = None
        self.ema_decay = ema_decay
        self.ema_steps = 0

    # ------------------------------------------------------------------ parameters / EMA
    def trainable_parameters(self):
        return list(self.unet.parameters()) + list(self.cond.parameters())

    def init_ema(self):
        self.ema_unet = deepcopy(self.unet).requires_grad_(False).eval()
        self.ema_cond = deepcopy(self.cond).requires_grad_(False).eval()

    @torch.no_grad()
    def update_ema(self):
        if self.ema_unet is None:
            return
        self.ema_steps += 1
        decay = min(self.ema_decay, (1.0 + self.ema_steps) / (10.0 + self.ema_steps))
        for ema_m, m in ((self.ema_unet, self.unet), (self.ema_cond, self.cond)):
            for ep, p in zip(ema_m.parameters(), m.parameters()):
                ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)

    def state_dicts(self):
        d = {
            'unet_state_dict': self.unet.state_dict(),
            'cond_state_dict': self.cond.state_dict(),
            'ema_steps': self.ema_steps,
        }
        if self.ema_unet is not None:
            d['ema_unet_state_dict'] = self.ema_unet.state_dict()
            d['ema_cond_state_dict'] = self.ema_cond.state_dict()
        return d

    def load_state_dicts(self, ckpt, strict=True):
        self.unet.load_state_dict(ckpt['unet_state_dict'], strict=strict)
        self.cond.load_state_dict(ckpt['cond_state_dict'], strict=strict)
        if 'ema_unet_state_dict' in ckpt:
            if self.ema_unet is None:
                self.init_ema()
            self.ema_unet.load_state_dict(ckpt['ema_unet_state_dict'], strict=strict)
            self.ema_cond.load_state_dict(ckpt['ema_cond_state_dict'], strict=strict)
        self.ema_steps = ckpt.get('ema_steps', self.ema_steps)

    def _nets(self, use_ema):
        if use_ema and self.ema_unet is not None:
            return self.ema_unet, self.ema_cond
        return self.unet, self.cond

    # ------------------------------------------------------------------ VAE helpers
    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.scaling_factor

    @torch.no_grad()
    def decode(self, latents):
        return self.vae.decode(latents / self.scaling_factor).sample

    # ------------------------------------------------------------------ training forward
    def forward(self, images, passage, cell_type, per_sample=False, use_ema=False):
        """
        Training / validation loss.
        In training mode (unet.training and not use_ema) CFG dropout and latent x0-L1 are applied.
        In eval mode (or use_ema=True) the plain epsilon-MSE is returned (clean validation metric).
        """
        unet, cond_net = self._nets(use_ema)
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=latents.device).long()
        noisy = self.train_scheduler.add_noise(latents, noise, timesteps)

        training = unet.training and not use_ema
        drop_mask = None
        if training and self.cfg_drop_prob > 0:
            drop_mask = torch.rand(latents.shape[0], device=latents.device) < self.cfg_drop_prob
        emb = cond_net(passage, cell_type, drop_mask)

        noise_pred = unet(noisy, timesteps, class_labels=emb).sample
        mse = F.mse_loss(noise_pred.float(), noise.float(), reduction='none').mean(dim=(1, 2, 3))

        if per_sample:
            return mse

        loss = mse.mean()
        if training and self.lpips_weight > 0:
            a = self.train_scheduler.alphas_cumprod.to(latents.device)[timesteps].view(-1, 1, 1, 1)
            pred_x0 = (noisy - (1 - a).sqrt() * noise_pred) / (a.sqrt() + 1e-8)
            loss = loss + self.lpips_weight * F.l1_loss(pred_x0.float(), latents.float())
        return loss

    # ------------------------------------------------------------------ inference
    def _cond_pair(self, cond_net, passage, cell_type):
        c = cond_net(passage, cell_type)
        u = cond_net(passage, cell_type, drop_mask=torch.ones_like(passage, dtype=torch.bool))
        return c, u

    def _denoise(self, unet, latents, timesteps, c_emb, u_emb, guidance_scale):
        B = latents.shape[0]
        for t in timesteps:
            t_batch = t.unsqueeze(0).repeat(B).to(latents.device)
            if guidance_scale > 1.0:
                eps_c = unet(latents, t_batch, class_labels=c_emb).sample
                eps_u = unet(latents, t_batch, class_labels=u_emb).sample
                eps = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = unet(latents, t_batch, class_labels=c_emb).sample
            latents = self.inference_scheduler.step(eps, t, latents).prev_sample
        return latents

    @torch.no_grad()
    def sample(self, passage, cell_type=None, noise=None, use_ema=True,
               guidance_scale=3.0, num_steps=50):
        """Unconditional-noise -> image generation at the given passage(s). Returns [0,1] images."""
        unet, cond_net = self._nets(use_ema)
        unet.eval(); cond_net.eval()
        device = next(unet.parameters()).device
        passage = torch.as_tensor(passage, dtype=torch.float32, device=device).reshape(-1)
        if cell_type is None:
            cell_type = torch.zeros_like(passage, dtype=torch.long)
        else:
            cell_type = torch.as_tensor(cell_type, dtype=torch.long, device=device).reshape(-1)
            if cell_type.numel() == 1 and passage.numel() > 1:
                cell_type = cell_type.expand(passage.numel()).clone()
        B = passage.shape[0]
        if noise is None:
            noise = torch.randn(B, 4, 64, 64, device=device)
        c_emb, u_emb = self._cond_pair(cond_net, passage, cell_type)
        self.inference_scheduler.set_timesteps(num_steps)
        latents = self._denoise(unet, noise, self.inference_scheduler.timesteps, c_emb, u_emb, guidance_scale)
        return (self.decode(latents).clamp(-1, 1) + 1) / 2

    @torch.no_grad()
    def translate(self, images, target_passage, cell_type=None, strength=0.6, num_steps=50,
                  use_ema=True, guidance_scale=3.0, noise=None):
        """SDEdit-style passage translation of real images to `target_passage`. Returns [0,1]."""
        unet, cond_net = self._nets(use_ema)
        unet.eval(); cond_net.eval()
        device = images.device
        B = images.shape[0]
        target_passage = torch.as_tensor(target_passage, dtype=torch.float32, device=device).reshape(-1)
        if target_passage.numel() == 1 and B > 1:
            target_passage = target_passage.expand(B).clone()
        if cell_type is None:
            cell_type = torch.zeros(B, dtype=torch.long, device=device)
        else:
            cell_type = torch.as_tensor(cell_type, dtype=torch.long, device=device).reshape(-1)
            if cell_type.numel() == 1 and B > 1:
                cell_type = cell_type.expand(B).clone()

        latents = self.encode(images)
        self.inference_scheduler.set_timesteps(num_steps)
        timesteps = self.inference_scheduler.timesteps
        start = int(len(timesteps) * (1 - strength))
        if noise is None:
            noise = torch.randn_like(latents)
        noisy = self.inference_scheduler.add_noise(latents, noise, timesteps[start])
        c_emb, u_emb = self._cond_pair(cond_net, target_passage, cell_type)
        out = self._denoise(unet, noisy, timesteps[start:], c_emb, u_emb, guidance_scale)
        return (self.decode(out).clamp(-1, 1) + 1) / 2
