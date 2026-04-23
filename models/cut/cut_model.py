"""
CUT Model: Contrastive Unpaired Translation.
Combines Generator + Discriminator + PatchNCE for stable unpaired image translation.
"""
import torch
import torch.nn as nn

from .generator import CUTGenerator
from .patchnce import PatchNCELoss, PatchMLPHead


class CUTModel(nn.Module):
    """
    Full CUT model for one direction of translation (e.g., young -> senescent).
    For bidirectional, instantiate two CUTModels.
    """
    def __init__(self, device='cuda', nce_layers=(0, 1, 2, 3),
                 nce_temperature=0.07, lambda_nce=1.0, lambda_idt=1.0):
        super().__init__()
        self.device = device
        self.nce_layers = nce_layers
        self.lambda_nce = lambda_nce
        self.lambda_idt = lambda_idt

        # Generator
        self.G = CUTGenerator(num_residual_blocks=9)

        # Discriminator (reuse CycleGAN's PatchGAN)
        from models.cyclegan.discriminator import Discriminator
        self.D = Discriminator()

        # PatchNCE loss + MLP heads for each encoder layer
        self.nce_loss_fn = PatchNCELoss(temperature=nce_temperature)
        self.mlp_heads = nn.ModuleList()
        for ch in [self.G.feat_channels[i] for i in nce_layers]:
            self.mlp_heads.append(PatchMLPHead(ch, 256))

    def forward(self, x):
        return self.G(x)

    def compute_gan_loss(self, pred, target_is_real):
        """LSGAN loss."""
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return nn.MSELoss()(pred, target)

    def compute_nce_loss(self, src, tgt):
        """
        Compute PatchNCE loss between source and target images.
        Extracts features from generator's encoder for both images.
        """
        # Get encoder features from source (input image)
        feat_src = self.G.encode(src)

        # Get encoder features from target (generated image passed through encoder)
        feat_tgt = self.G.encode(tgt)

        total_nce = 0.0
        for i, layer_idx in enumerate(self.nce_layers):
            # Project features through MLP
            f_src = feat_src[layer_idx]
            f_tgt = feat_tgt[layer_idx]

            B, C, H, W = f_src.shape
            num_patches = H * W

            # Flatten spatial dims
            f_src_flat = f_src.permute(0, 2, 3, 1).reshape(B, num_patches, C)
            f_tgt_flat = f_tgt.permute(0, 2, 3, 1).reshape(B, num_patches, C)

            # Sample 256 patches
            if num_patches > 256:
                sample_ids = torch.randperm(num_patches, device=self.device)[:256]
                f_src_sampled = f_src_flat[:, sample_ids, :]
                f_tgt_sampled = f_tgt_flat[:, sample_ids, :]
            else:
                f_src_sampled = f_src_flat
                f_tgt_sampled = f_tgt_flat

            # Pass only the sampled patches through the MLP (Huge speedup!)
            # Shape: [B, 256, C] -> [B, 256, 256]
            f_src_proj = self.mlp_heads[i](f_src_sampled)
            f_tgt_proj = self.mlp_heads[i](f_tgt_sampled)

            total_nce += self.nce_loss_fn(f_tgt_proj, f_src_proj)

        return total_nce / len(self.nce_layers)

    def train_step(self, real_src, real_tgt):
        """
        One training step.
        Args:
            real_src: source domain image (e.g., young)
            real_tgt: target domain image (e.g., senescent)
        Returns:
            dict of losses
        """
        # ==================
        # Generate fake
        # ==================
        fake_tgt = self.G(real_src)

        # ==================
        # Discriminator loss
        # ==================
        # Real
        pred_real = self.D(real_tgt)
        loss_D_real = self.compute_gan_loss(pred_real, True)

        # Fake (detached)
        pred_fake = self.D(fake_tgt.detach())
        loss_D_fake = self.compute_gan_loss(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        # ==================
        # Generator loss
        # ==================
        # GAN loss
        pred_fake_for_G = self.D(fake_tgt)
        loss_G_gan = self.compute_gan_loss(pred_fake_for_G, True)

        # PatchNCE loss (correspondence between src and fake_tgt)
        loss_G_nce = self.compute_nce_loss(real_src, fake_tgt) * self.lambda_nce

        # Identity NCE: feed target domain through G, NCE should be low
        if self.lambda_idt > 0:
            idt_tgt = self.G(real_tgt)
            loss_G_idt = self.compute_nce_loss(real_tgt, idt_tgt) * self.lambda_idt
        else:
            loss_G_idt = torch.tensor(0.0, device=self.device)

        loss_G = loss_G_gan + loss_G_nce + loss_G_idt

        return {
            'loss_D': loss_D,
            'loss_G': loss_G,
            'loss_G_gan': loss_G_gan,
            'loss_G_nce': loss_G_nce,
            'loss_G_idt': loss_G_idt,
            'fake_tgt': fake_tgt,
        }
