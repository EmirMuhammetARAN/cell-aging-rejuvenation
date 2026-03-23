import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentDiscriminator(nn.Module):
    def __init__(self, in_channels=4, num_classes=2, ndf=64):
        """
        A discriminator that operates directly on the VAE latents (4 channels, 64x64).
        It uses Spectral Normalization for stability and Projection Conditioning 
        for class labels (Young vs Senescent).
        """
        super().__init__()
        
        # 4 x 64 x 64 -> ndf x 32 x 32
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1))
        
        # ndf x 32 x 32 -> (ndf*2) x 16 x 16
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1))
        
        # (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
        self.conv3 = nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1))
        
        # (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
        self.conv4 = nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1))
        
        # (ndf*8) x 4 x 4 -> 1 x 1 x 1 (unconditional score)
        self.conv5 = nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0))
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        # Class embedding for Projection Discriminator
        self.embed = nn.utils.spectral_norm(nn.Embedding(num_classes, ndf * 8))
        
    def forward(self, x, labels=None):
        # Forward pass through conv layers
        h = self.leaky_relu(self.conv1(x))
        h = self.leaky_relu(self.conv2(h))
        h = self.leaky_relu(self.conv3(h))
        h = self.leaky_relu(self.conv4(h))  # Shape: [B, ndf*8, 4, 4]
        
        # Unconditional score
        out = self.conv5(h).view(-1, 1)     # Shape: [B, 1]
        
        # Conditional score (Projection Discriminator)
        if labels is not None:
            # Global sum pooling over spatial dimensions to match embedding dimension
            h_pooled = torch.sum(h, dim=(2, 3))  # Shape: [B, ndf*8]
            
            # Get class embedding
            y_emb = self.embed(labels)           # Shape: [B, ndf*8]
            
            # Dot product for projection
            proj_score = torch.sum(h_pooled * y_emb, dim=1, keepdim=True)  # Shape: [B, 1]
            out = out + proj_score
            
        return out
