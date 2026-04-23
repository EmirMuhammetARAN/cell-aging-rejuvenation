"""
PatchNCE Loss for CUT (Contrastive Unpaired Translation).
Implements InfoNCE contrastive loss on multi-layer encoder patches.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMLPHead(nn.Module):
    """Small 2-layer MLP to project encoder features for contrastive learning."""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)


class PatchNCELoss(nn.Module):
    """
    PatchNCE Loss: for each spatial location in the generated image,
    the corresponding location in the input is 'positive' and
    all other locations are 'negatives'.
    
    Temperature-scaled InfoNCE (cross-entropy over cosine similarities).
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, feat_q, feat_k):
        """
        Args:
            feat_q: query features from generated image [B, N, C]
            feat_k: key features from input image [B, N, C]
        Returns:
            NCE loss scalar
        """
        B, N, C = feat_q.shape

        # L2 normalize
        feat_q = F.normalize(feat_q, dim=-1)
        feat_k = F.normalize(feat_k, dim=-1)

        # Positive: dot product between corresponding patches
        # [B, N, 1]
        l_pos = (feat_q * feat_k).sum(dim=-1, keepdim=True)

        # Negative: dot product between query and ALL sampled key patches
        # [B, N, N]
        l_neg = torch.bmm(feat_q, feat_k.permute(0, 2, 1))

        # Logits: [B, N, 1+N]
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temperature

        # Labels: positive is always index 0
        labels = torch.zeros(B * N, dtype=torch.long, device=feat_q.device)
        logits = logits.reshape(-1, logits.shape[-1])

        loss = self.ce_loss(logits, labels).mean()
        return loss
