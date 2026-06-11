import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SameShotAttention(nn.Module):
    """Attention among active entities within the same shot."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, presence: torch.Tensor,
                relation_bias: torch.Tensor = None) -> torch.Tensor:
        """
        q: [B, S, E, d]
        presence: [B, S, E]   (0/1)
        relation_bias: [B, S, num_heads, E, E]
        """
        B, S, E, d = q.shape
        H, dh = self.num_heads, self.d_head

        Q = self.q_proj(q).view(B, S, E, H, dh).permute(0, 1, 3, 2, 4)  # [B,S,H,E,dh]
        K = self.k_proj(q).view(B, S, E, H, dh).permute(0, 1, 3, 2, 4)
        V = self.v_proj(q).view(B, S, E, H, dh).permute(0, 1, 3, 2, 4)

        scores = (Q @ K.transpose(-2, -1)) / self.scale  # [B,S,H,E,E]

        # Presence mask: only attend to active entities (presence==1)
        # mask[b,s,e_prime] = 0 if active, -inf if not
        attn_mask = (1 - presence.float()).unsqueeze(2) * -1e9  # [B,S,1,E]
        attn_mask = attn_mask.unsqueeze(-2).expand_as(scores)  # [B,S,H,E,E]

        scores = scores + attn_mask

        if relation_bias is not None:
            scores = scores + relation_bias

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).permute(0, 1, 3, 2, 4).reshape(B, S, E, d)  # [B,S,E,d]
        return self.out_proj(out)


class SameEntityTemporalAttention(nn.Module):
    """Attention across shots for the same entity."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.scale = math.sqrt(self.d_head)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, presence: torch.Tensor) -> torch.Tensor:
        """
        q: [B, S, E, d]
        presence: [B, S, E]
        """
        B, S, E, d = q.shape
        H, dh = self.num_heads, self.d_head

        # Transpose to [B, E, S, d] for entity-wise processing
        q_t = q.permute(0, 2, 1, 3)  # [B, E, S, d]
        pres_t = presence.permute(0, 2, 1)  # [B, E, S]

        Q = self.q_proj(q_t).view(B, E, S, H, dh).permute(0, 1, 3, 2, 4)  # [B,E,H,S,dh]
        K = self.k_proj(q_t).view(B, E, S, H, dh).permute(0, 1, 3, 2, 4)
        V = self.v_proj(q_t).view(B, E, S, H, dh).permute(0, 1, 3, 2, 4)

        scores = (Q @ K.transpose(-2, -1)) / self.scale  # [B,E,H,S,S]

        # mask: only attend to shots where entity is active
        attn_mask = (1 - pres_t.float()).unsqueeze(2)  # [B,E,1,S]
        attn_mask = attn_mask.unsqueeze(-2).expand_as(scores) * -1e9  # [B,E,H,S,S]
        scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ V).permute(0, 1, 3, 2, 4).reshape(B, E, S, d)
        out = out.permute(0, 2, 1, 3)  # [B, S, E, d]
        return self.out_proj(out)


class LayoutBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_shot_attn: bool = True, use_temporal_attn: bool = True):
        super().__init__()
        self.use_shot_attn = use_shot_attn
        self.use_temporal_attn = use_temporal_attn
        if use_shot_attn:
            self.shot_attn = SameShotAttention(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
        if use_temporal_attn:
            self.temp_attn = SameEntityTemporalAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, presence: torch.Tensor,
                relation_bias: torch.Tensor = None) -> torch.Tensor:
        if self.use_shot_attn:
            q = self.norm1(q + self.shot_attn(q, presence, relation_bias))
        if self.use_temporal_attn:
            q = self.norm2(q + self.temp_attn(q, presence))
        q = self.norm3(q + self.ffn(q))
        return q
