"""
ResidualEntityConditioner: identity residual projector for entity memory.

Key design change vs EntityMemoryConditioner:
  - Old: conditioner(v_e) ≈ text_enc(e)  → doubles text direction, no new info
  - New: conditioner(v_e) = r_e  where r_e ⊥ text_enc(e)

Grounding at inference:
  g_e = text_enc(e) + λ_m * r_e
       [what is "cat"] + [which specific cat this is]

r_e is trained to be:
  1. Close for same entity (InfoNCE / SupCon)
  2. Far for different entities
  3. Orthogonal to text embedding (text-orth regularization)
  4. Unit-norm (norm regularization)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualEntityConditioner(nn.Module):
    """
    Projects CLIP ViT-H/14 visual embedding (1024d) to identity residual space (768d).

    During training: forward(x) returns raw residual.
    During inference: use orthogonalized(x, text_emb) to remove text direction.
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
        """Unit-norm residual: (B, 1024) → (B, 768), L2-normalized."""
        h = self.norm(self.proj(x))
        return F.normalize(h, dim=-1)   # unit hypersphere — consistent scale at inference

    @staticmethod
    def orthogonalize(r: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Remove text direction from r.
        r ← r - proj_{text_emb}(r)

        Args:
            r:        (B, D) or (D,)  residual embeddings
            text_emb: (B, D) or (D,)  text encoder pooler_output
        Returns:
            orthogonalized residual, same shape as r
        """
        t_n = F.normalize(text_emb, dim=-1)
        projection = (r * t_n).sum(-1, keepdim=True) * t_n
        return r - projection


# ── Loss ─────────────────────────────────────────────────────────────────────

def supcon_loss(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    entity_ids: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    SupCon loss over paired embeddings.

    r_a, r_b: (B, D) — two views of same entity (different image augmentations)
    entity_ids: (B,)  — integer entity label (same entity → same id)

    All 2B embeddings form the pool; positives = same entity_id, different index.
    """
    B = r_a.shape[0]
    all_r = F.normalize(torch.cat([r_a, r_b], dim=0), dim=-1)   # (2B, D)
    ids   = torch.cat([entity_ids, entity_ids], dim=0)           # (2B,)

    sim = torch.mm(all_r, all_r.T) / temperature                 # (2B, 2B)

    self_mask = torch.eye(2 * B, device=r_a.device, dtype=torch.bool)
    pos_mask  = (ids.unsqueeze(0) == ids.unsqueeze(1)) & ~self_mask  # (2B, 2B)

    # Mask self from denominator
    sim_masked = sim.masked_fill(self_mask, -1e9)
    log_denom  = torch.logsumexp(sim_masked, dim=1)              # (2B,)

    # SupCon: average over positives
    n_pos = pos_mask.sum(1).clamp(min=1).float()
    loss  = -(sim_masked * pos_mask.float()).sum(1) / n_pos + log_denom
    return loss.mean()


def text_orth_loss(r: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
    """
    Penalise cosine alignment between residual and text direction.
    L_orth = |cos(r, text_emb)|
    """
    r_n = F.normalize(r, dim=-1)
    t_n = F.normalize(text_emb, dim=-1)
    return (r_n * t_n).sum(-1).abs().mean()


def norm_reg_loss(r: torch.Tensor, target_norm: float = 1.0) -> torch.Tensor:
    """L_norm = (||r|| - ρ)²  prevents residual collapse or explosion."""
    return ((r.norm(dim=-1) - target_norm) ** 2).mean()


def residual_conditioner_loss(
    r_a: torch.Tensor,
    r_b: torch.Tensor,
    entity_ids: torch.Tensor,
    text_embs: torch.Tensor,
    temperature: float = 0.07,
    lambda_orth: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss for ResidualEntityConditioner.

    norm_reg is removed: model.forward() outputs unit-norm vectors (L2-normalized),
    so no explicit norm regularization is needed.

    Args:
        r_a, r_b:    (B, D) unit-norm residuals from two views of same entity
        entity_ids:  (B,)   integer entity labels
        text_embs:   (B, D) text encoder pooler_output for each entity
    Returns:
        total loss, dict of component losses
    """
    l_cont = supcon_loss(r_a, r_b, entity_ids, temperature)
    l_orth = (text_orth_loss(r_a, text_embs) + text_orth_loss(r_b, text_embs)) / 2

    total = l_cont + lambda_orth * l_orth
    return total, {
        "contrastive": l_cont.item(),
        "orth":        l_orth.item(),
    }
