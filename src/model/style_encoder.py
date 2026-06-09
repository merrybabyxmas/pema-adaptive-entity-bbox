"""
Phase 2: GlobalStyleTokenEncoder

Encodes a style reference image into K_g style tokens for GLIGEN conditioning.

Architecture:
  style_ref → CLIP ViT-H/14 (frozen) → StyleEncoder (MLP) → G ∈ R^{K_g × D}

K_g style tokens condition the diffusion globally (full-image bbox, small γ).
They encode video-wide: color palette, lighting, rendering style, texture.
They must NOT encode entity-specific content (handled by entity memory).

Usage (MVP, untrained):
  style_enc = StyleEncoder().to(device)
  clip_emb  = entity_encoder.encode(bg_image)         # (1024,)
  G_style   = style_enc(clip_emb.unsqueeze(0))         # (1, K_g, 768)

Training (Phase 2 full):
  SimCLR on background-only image crops.
  Two spatial crops of same image → positive pair (same style).
  Images from different scenes → negative.
  Loss: InfoNCE on mean-pooled style tokens.
  See: scripts/train_style_encoder.py
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    """
    Projects CLIP ViT-H/14 embedding (1024d) to K_g style tokens (grounding_dim=768d each).

    n_tokens=4 style tokens, each capturing a different style facet.
    Style tokens are injected as GLIGEN grounding phrases with bbox=[0,0,1,1].
    Use small style_weight (γ_style=0.1~0.3) to avoid overwhelming entity grounding.
    """

    def __init__(
        self,
        clip_dim: int = 1024,
        grounding_dim: int = 768,
        n_tokens: int = 4,
        hidden_dim: int = 1024,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_tokens    = n_tokens
        self.grounding_dim = grounding_dim

        # Shared trunk: compress + mix CLIP features
        self.trunk = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # K_g independent heads: each projects to a style token
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, grounding_dim)
            for _ in range(n_tokens)
        ])

        # Per-token LayerNorm (applied to each K output)
        self.norms = nn.ModuleList([
            nn.LayerNorm(grounding_dim)
            for _ in range(n_tokens)
        ])

    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_emb: (B, 1024) or (1024,) CLIP ViT-H/14 pooled image embedding
        Returns:
            style_tokens: (B, K_g, grounding_dim)  — K_g style conditioning tokens
        """
        if clip_emb.dim() == 1:
            clip_emb = clip_emb.unsqueeze(0)

        h = self.trunk(clip_emb)                                    # (B, hidden_dim)
        tokens = torch.stack(
            [norm(head(h)) for head, norm in zip(self.heads, self.norms)],
            dim=1,
        )                                                            # (B, K_g, grounding_dim)
        return tokens

    def aggregate(self, style_tokens: torch.Tensor) -> torch.Tensor:
        """Mean-pool K_g tokens to a single embedding for contrastive loss."""
        return style_tokens.mean(dim=1)                             # (B, D)


# ── Style consistency loss (Phase 2 full training) ────────────────────────────

def style_consistency_loss(
    s_a: torch.Tensor,
    s_b: torch.Tensor,
    scene_ids: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    InfoNCE on aggregated style tokens.

    s_a, s_b:   (B, D) aggregated style embeddings (mean of K_g tokens)
    scene_ids:  (B,)   integer scene/image label — same scene → positive pair
    """
    B = s_a.shape[0]
    all_s = F.normalize(torch.cat([s_a, s_b], dim=0), dim=-1)  # (2B, D)
    ids   = torch.cat([scene_ids, scene_ids], dim=0)            # (2B,)

    sim      = torch.mm(all_s, all_s.T) / temperature           # (2B, 2B)
    self_m   = torch.eye(2 * B, device=s_a.device, dtype=torch.bool)
    pos_mask = (ids.unsqueeze(0) == ids.unsqueeze(1)) & ~self_m

    sim_masked  = sim.masked_fill(self_m, -1e9)
    log_denom   = torch.logsumexp(sim_masked, dim=1)
    n_pos       = pos_mask.sum(1).clamp(min=1).float()
    loss        = -(sim_masked * pos_mask.float()).sum(1) / n_pos + log_denom
    return loss.mean()
