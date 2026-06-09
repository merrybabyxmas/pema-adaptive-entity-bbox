import torch
import torch.nn as nn
import open_clip

from src.data.schema import STATE_VOCAB, RELATION_VOCAB


class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai", freeze=True):
        super().__init__()
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.d_out = model.text_projection.shape[1] if hasattr(model, 'text_projection') and model.text_projection is not None else 512
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode_texts(self, texts: list[str], device) -> torch.Tensor:
        """texts: list of strings -> [N, d_out]"""
        tokens = self.tokenizer(texts).to(device)
        feats = self.model.encode_text(tokens)
        feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-6)
        return feats.float()

    def encode_batch_shots(self, shot_prompts_batch: list[list[str]], device) -> torch.Tensor:
        """shot_prompts_batch: [B, S] strings -> [B, S, d_out]"""
        B = len(shot_prompts_batch)
        S = len(shot_prompts_batch[0])
        flat = [p for prompts in shot_prompts_batch for p in prompts]
        emb = self.encode_texts(flat, device)  # [B*S, d]
        return emb.view(B, S, -1)

    def encode_batch_entities(self, entity_names_batch: list[list[str]], device) -> torch.Tensor:
        """entity_names_batch: [B, E] -> [B, E, d_out]"""
        B = len(entity_names_batch)
        E = len(entity_names_batch[0])
        flat = [e if e else "object" for names in entity_names_batch for e in names]
        emb = self.encode_texts(flat, device)  # [B*E, d]
        return emb.view(B, E, -1)


class StateEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(len(STATE_VOCAB), d_model)

    def forward(self, state_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(state_ids)


class RelationBias(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.emb = nn.Embedding(len(RELATION_VOCAB), num_heads)

    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """relation_ids: [B, S, E, E] -> bias: [B, S, num_heads, E, E]"""
        bias = self.emb(relation_ids)  # [B, S, E, E, num_heads]
        return bias.permute(0, 1, 4, 2, 3)  # [B, S, num_heads, E, E]
