import torch
import torch.nn as nn

from src.model.attention import LayoutBlock
from src.model.heads import BBoxHead
from src.model.embeddings import StateEmbedding, RelationBias


class PresenceAwareBBoxPlanner(nn.Module):
    def __init__(self, d_text: int = 512, d_model: int = 256,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.shot_proj = nn.Linear(d_text, d_model)
        self.entity_proj = nn.Linear(d_text, d_model)
        self.state_emb = StateEmbedding(d_model)
        self.relation_bias = RelationBias(num_heads)

        self.layers = nn.ModuleList([
            LayoutBlock(d_model, num_heads, dropout) for _ in range(num_layers)
        ])
        self.bbox_head = BBoxHead(d_model)

    def forward(self, shot_emb: torch.Tensor, entity_emb: torch.Tensor,
                state_ids: torch.Tensor, presence: torch.Tensor,
                relation_ids: torch.Tensor) -> torch.Tensor:
        """
        shot_emb:    [B, S, d_text]
        entity_emb:  [B, E, d_text]
        state_ids:   [B, S, E]
        presence:    [B, S, E]   int 0/1
        relation_ids:[B, S, E, E]
        -> pred_boxes: [B, S, E, 4] in [0,1] cxcywh
        """
        B, S, E = presence.shape

        hs = self.shot_proj(shot_emb)[:, :, None, :]      # [B,S,1,d]
        he = self.entity_proj(entity_emb)[:, None, :, :]  # [B,1,E,d]
        hstate = self.state_emb(state_ids)                 # [B,S,E,d]

        q = hs + he + hstate                               # [B,S,E,d]

        rel_bias = self.relation_bias(relation_ids)        # [B,S,H,E,E]

        for layer in self.layers:
            q = layer(q, presence, rel_bias)

        boxes = self.bbox_head(q)                          # [B,S,E,4]
        return boxes


class MLPBaseline(nn.Module):
    """Independent MLP baseline — no attention."""
    def __init__(self, d_text: int = 512, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.shot_proj = nn.Linear(d_text, d_model)
        self.entity_proj = nn.Linear(d_text, d_model)
        self.state_emb = StateEmbedding(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, shot_emb, entity_emb, state_ids, presence, relation_ids):
        B, S, E = presence.shape
        hs = self.shot_proj(shot_emb)[:, :, None, :].expand(B, S, E, -1)
        he = self.entity_proj(entity_emb)[:, None, :, :].expand(B, S, E, -1)
        hst = self.state_emb(state_ids)
        q = torch.cat([hs, he, hst], dim=-1)
        return torch.sigmoid(self.mlp(q))


def build_model(cfg: dict) -> nn.Module:
    arch = cfg.get("arch", "planner")
    d_text = cfg.get("d_text", 512)
    d_model = cfg.get("d_model", 256)
    num_layers = cfg.get("num_layers", 4)
    num_heads = cfg.get("num_heads", 8)
    dropout = cfg.get("dropout", 0.1)

    if arch == "mlp_baseline":
        return MLPBaseline(d_text=d_text, d_model=d_model, dropout=dropout)
    return PresenceAwareBBoxPlanner(
        d_text=d_text, d_model=d_model,
        num_layers=num_layers, num_heads=num_heads, dropout=dropout,
    )
