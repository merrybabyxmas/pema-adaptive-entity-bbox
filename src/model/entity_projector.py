"""
EntityProjector: lightweight MLP trained with triplet contrastive loss.

Maps CLIP ViT-H/14 (1024d) entity embeddings to an identity-preserving
subspace. At inference the output is fed through IP-Adapter's ImageProjection
layer to produce cross-attention tokens (4 × 768).

Architecture: 3-layer residual MLP (1024 → 2048 → 1024 → 1024).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityProjector(nn.Module):
    def __init__(self, dim: int = 1024, hidden: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1024) → (B, 1024) enhanced embedding."""
        return self.norm(x + self.net(x))


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor,
                 negative: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """Triplet loss in cosine similarity space."""
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positive, dim=-1)
    n = F.normalize(negative, dim=-1)
    pos_sim = (a * p).sum(-1)       # (B,)
    neg_sim = (a * n).sum(-1)       # (B,)
    loss = F.relu(neg_sim - pos_sim + margin)
    return loss.mean()


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor,
                     labels: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    labels=1 → same entity (should be similar),
    labels=0 → different entity (should be distant).
    """
    cos = F.cosine_similarity(z1, z2, dim=-1)
    pos_loss = labels * (1 - cos) ** 2
    neg_loss = (1 - labels) * F.relu(cos - margin) ** 2
    return (pos_loss + neg_loss).mean()
