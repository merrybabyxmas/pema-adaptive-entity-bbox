"""
EntityMemoryConditioner: projects CLIP ViT-H/14 embeddings (1024d)
into GLIGEN's grounding dimension (768d = SD1.5 cross_attention_dim).

Training objective: align image memory embedding with text encoder
pooler_output for the entity phrase, so entity memory can replace
(or augment) GLIGEN's text grounding tokens.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityMemoryConditioner(nn.Module):
    """
    Projects entity CLIP ViT-H/14 embeddings (1024d) to GLIGEN grounding
    space (768d), optionally fusing with text encoder output.

    Usage at inference:
        cond_emb = conditioner(clip_emb_1024)   # → (768,)
        grounding_emb = text_emb_768 + cond_emb  # augment text with memory
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 768,
                 hidden_dim: int = 896, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., 1024) → (..., 768)"""
        return self.norm(self.proj(x))


def conditioner_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Align conditioner output to text encoder pooler_output.
    pred:   (B, 768) from EntityMemoryConditioner
    target: (B, 768) from CLIPTextModel.pooler_output for entity phrase
    """
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    cos_loss = 1.0 - (pred_n * tgt_n).sum(-1).mean()
    mse_loss = F.mse_loss(pred, target)
    return cos_loss + 0.1 * mse_loss
