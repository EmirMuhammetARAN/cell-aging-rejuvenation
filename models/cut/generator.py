"""
CUT Generator: ResNet-based generator with multi-layer feature extraction.
Based on the same architecture as CycleGAN but returns intermediate encoder 
features for PatchNCE contrastive loss.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class CUTGenerator(nn.Module):
    """
    ResNet generator that can optionally return encoder features
    at multiple layers for PatchNCE loss computation.
    """
    def __init__(self, num_residual_blocks=9):
        super().__init__()

        # Encoder layers (we extract features from these)
        self.enc1 = nn.Sequential(  # 3 -> 64, full res
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(  # 64 -> 128, /2
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(  # 128 -> 256, /4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(num_residual_blocks)]
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        )

        # Feature channels at each extraction layer
        self.feat_channels = [64, 128, 256, 256]  # enc1, enc2, enc3, res_out

    def forward(self, x, return_feats=False):
        """
        Args:
            x: input image [B, 3, H, W]
            return_feats: if True, also return encoder features for NCE loss
        Returns:
            output image, and optionally list of encoder feature maps
        """
        feats = []

        e1 = self.enc1(x)
        if return_feats:
            feats.append(e1)

        e2 = self.enc2(e1)
        if return_feats:
            feats.append(e2)

        e3 = self.enc3(e2)
        if return_feats:
            feats.append(e3)

        r = self.res_blocks(e3)
        if return_feats:
            feats.append(r)

        d1 = self.dec1(r)
        d2 = self.dec2(d1)
        out = self.final(d2)

        if return_feats:
            return out, feats
        return out

    def encode(self, x):
        """Extract only encoder features (for computing NCE on input images)."""
        feats = []
        e1 = self.enc1(x)
        feats.append(e1)
        e2 = self.enc2(e1)
        feats.append(e2)
        e3 = self.enc3(e2)
        feats.append(e3)
        r = self.res_blocks(e3)
        feats.append(r)
        return feats
