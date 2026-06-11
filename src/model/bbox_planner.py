import torch
import torch.nn as nn

from src.model.attention import LayoutBlock
from src.model.heads import BBoxHead, MLP
from src.model.embeddings import StateEmbedding, RelationBias


class PresenceAwareBBoxPlanner(nn.Module):
    def __init__(self, d_text: int = 512, d_model: int = 256,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1,
                 use_state: bool = True, use_relation: bool = True,
                 use_shot_attn: bool = True, use_temporal_attn: bool = True,
                 use_shot_emb: bool = True, use_entity_emb: bool = True):
        super().__init__()
        self.use_state = use_state
        self.use_relation = use_relation
        self.use_shot_emb = use_shot_emb
        self.use_entity_emb = use_entity_emb
        self.shot_proj = nn.Linear(d_text, d_model)
        self.entity_proj = nn.Linear(d_text, d_model)
        if use_state:
            self.state_emb = StateEmbedding(d_model)
        if use_relation:
            self.relation_bias = RelationBias(num_heads)

        self.layers = nn.ModuleList([
            LayoutBlock(d_model, num_heads, dropout,
                        use_shot_attn=use_shot_attn, use_temporal_attn=use_temporal_attn)
            for _ in range(num_layers)
        ])
        self.bbox_head = BBoxHead(d_model)
        # occlusion depth per entity (0=back, 1=front), sigmoid in [0,1]
        self.depth_head = MLP(d_model, d_model, 1, num_layers=2)

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

        q = torch.zeros(B, S, E, hs.shape[-1], device=hs.device, dtype=hs.dtype)
        if self.use_shot_emb:
            q = q + hs                                     # broadcast [B,S,1,d]
        if self.use_entity_emb:
            q = q + he                                     # broadcast [B,1,E,d]
        if self.use_state:
            q = q + self.state_emb(state_ids)              # [B,S,E,d]

        rel_bias = self.relation_bias(relation_ids) if self.use_relation else None

        for layer in self.layers:
            q = layer(q, presence, rel_bias)

        boxes = self.bbox_head(q)                          # [B,S,E,4]
        depth = torch.sigmoid(self.depth_head(q))          # [B,S,E,1] in [0,1]
        return torch.cat([boxes, depth], dim=-1)           # [B,S,E,5]


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
            nn.Linear(d_model, 5),
        )

    def forward(self, shot_emb, entity_emb, state_ids, presence, relation_ids):
        B, S, E = presence.shape
        hs = self.shot_proj(shot_emb)[:, :, None, :].expand(B, S, E, -1)
        he = self.entity_proj(entity_emb)[:, None, :, :].expand(B, S, E, -1)
        hst = self.state_emb(state_ids)
        q = torch.cat([hs, he, hst], dim=-1)
        return torch.sigmoid(self.mlp(q))  # [B,S,E,5], last channel = depth


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
        use_state=cfg.get("use_state", True),
        use_relation=cfg.get("use_relation", True),
        use_shot_attn=cfg.get("use_shot_attn", True),
        use_temporal_attn=cfg.get("use_temporal_attn", True),
        use_shot_emb=cfg.get("use_shot_emb", True),
        use_entity_emb=cfg.get("use_entity_emb", True),
    )
