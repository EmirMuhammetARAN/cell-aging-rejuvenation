import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ldm.model_lpips import CellLDM

class CellLDM_v14(CellLDM):
    """
    LDM v14: Mask-Guided / Region-Focused Fine-Tuning
    Inherits CellLDM (v12 architecture) and applies spatial mask re-weighting
    to both Diffusion Noise MSE Loss and Latent Structural L1 Loss.
    
    Cell foreground (M_dilated) receives full loss weight (1.0),
    while the background receives stabilizer weight bg_weight (default 0.05).
    """
    def __init__(self, lpips_weight=0.1, bg_weight=0.05, **kwargs):
        super().__init__(lpips_weight=lpips_weight, **kwargs)
        self.bg_weight = bg_weight

    def forward(self, images, labels=None, masks=None):
        """
        images: [B, 3, 512, 512] in [-1, 1]
        labels: [B] class labels (0: Young, 1: Senescent)
        masks:  [B, 1, 64, 64] float binary or soft mask in latent space
        """
        latents = self.encode(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.train_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device
        ).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)
        
        # Classifier-Free Guidance dropout during training
        if labels is not None and self.training:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.cfg_drop_prob
            labels = labels.clone()
            labels[drop_mask] = self.num_classes  

        noise_pred = self.unet(noisy_latents, timesteps, class_labels=labels).sample
        
        # 1. Mask-Weighted Diffusion MSE Loss
        diff_sq = (noise_pred - noise) ** 2
        if masks is not None:
            m_cell = masks.clamp(0.0, 1.0)
            m_bg = 1.0 - m_cell
            n_cell = m_cell.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
            n_bg = m_bg.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
            
            loss_cell = ((diff_sq * m_cell).sum(dim=(1, 2, 3), keepdim=True) / n_cell).mean()
            loss_bg = ((diff_sq * m_bg).sum(dim=(1, 2, 3), keepdim=True) / n_bg).mean()
            mse_loss = loss_cell + self.bg_weight * loss_bg
        else:
            mse_loss = diff_sq.mean()

        # 2. Mask-Weighted Latent Structural Loss (Latent L1)
        if self.lpips_weight > 0 and self.training:
            alpha_prod_t = self.train_scheduler.alphas_cumprod.to(latents.device)[timesteps]
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)
            pred_x0 = (noisy_latents - (1 - alpha_prod_t).sqrt() * noise_pred) / (alpha_prod_t.sqrt() + 1e-8)
            latent_diff = (pred_x0 - latents).abs()
            
            if masks is not None:
                latent_loss_cell = ((latent_diff * m_cell).sum(dim=(1, 2, 3), keepdim=True) / n_cell).mean()
                latent_loss_bg = ((latent_diff * m_bg).sum(dim=(1, 2, 3), keepdim=True) / n_bg).mean()
                latent_loss = latent_loss_cell + self.bg_weight * latent_loss_bg
            else:
                latent_loss = latent_diff.mean()

            return mse_loss + self.lpips_weight * latent_loss
        
        return mse_loss
