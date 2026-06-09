"""
Phase 4: Entity/Style Branch Adapter

Adds two new cross-attention branches to every SD 1.5 attn2 layer:

  z_text   = Attn(Q, K_text, V_text)                       [text, full image]
  z_entity = Attn(Q, K_ent, V_ent, mask=bbox_mask)         [entity, bbox-localized]
  z_style  = Attn(Q, K_style, V_style)                     [style, full image]
  z = (z_text + γ_e * z_entity + γ_g * z_style) → to_out

Trainable parameters (UNet frozen):
  Per attn2 layer:
    entity_to_k, entity_to_v : Linear(1024 → inner_dim)
    style_to_k,  style_to_v  : Linear(768  → inner_dim)
    gamma_entity, gamma_style : scalar (init 0.1, 0.05)

Bbox masking:
  Spatial positions in [H_feat × W_feat] outside entity's bbox get
  attention logit −inf → entity memory conditions only its region.

Usage:
  adapter = EntityStyleAdapter(unet)
  adapter.register_to_unet()           # replace attn2 processors
  adapter.set_conditions(entity_tokens, style_tokens, entity_bboxes)
  output = unet(...)                   # adapter active
  adapter.clear_conditions()
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── BBox spatial mask ─────────────────────────────────────────────────────────

def make_bbox_attn_bias(
    entity_bboxes: list[list[float]],
    H: int, W: int,
    n_heads: int,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build attention bias for entity cross-attention.
    Positions outside an entity's bbox get -1e9 (effectively -inf).

    Returns: (B, n_heads, H*W, n_entities)
    """
    n_ent = len(entity_bboxes)
    HW = H * W
    bias = torch.zeros(B, n_heads, HW, n_ent, device=device, dtype=dtype)

    ys = (torch.arange(H, device=device).float() + 0.5) / H  # (H,)
    xs = (torch.arange(W, device=device).float() + 0.5) / W  # (W,)

    for e_idx, bbox in enumerate(entity_bboxes):
        x1, y1, x2, y2 = bbox
        in_bbox = (
            (ys >= y1).unsqueeze(1) & (ys < y2).unsqueeze(1) &
            (xs >= x1).unsqueeze(0) & (xs < x2).unsqueeze(0)
        )  # (H, W) bool
        outside = ~in_bbox.view(HW)  # (HW,)
        bias[:, :, outside, e_idx] = torch.finfo(dtype).min / 2

    return bias


# ── Per-layer adapter processor ───────────────────────────────────────────────

class EntityStyleAdapterProcessor(nn.Module):
    """
    Replaces the AttnProcessor on a single attn2 (cross-attention) layer.

    Entity and style branches share Q with the text branch but use their own
    K/V projections.  All three branches are combined before to_out:

        z_combined = z_text + γ_e * z_entity + γ_g * z_style
        output = to_out(z_combined)

    γ_e, γ_g initialized to non-zero (0.1, 0.05) so K/V projections
    always receive gradients during training.
    """

    def __init__(
        self,
        inner_dim: int,
        entity_dim: int = 1024,
        style_dim: int = 768,
        gamma_entity_init: float = 0.1,
        gamma_style_init: float = 0.05,
    ):
        super().__init__()
        self.inner_dim = inner_dim

        # Entity branch K/V projections
        self.entity_to_k = nn.Linear(entity_dim, inner_dim, bias=False)
        self.entity_to_v = nn.Linear(entity_dim, inner_dim, bias=False)

        # Style branch K/V projections
        self.style_to_k = nn.Linear(style_dim, inner_dim, bias=False)
        self.style_to_v = nn.Linear(style_dim, inner_dim, bias=False)

        # Learnable gates
        self.gamma_entity = nn.Parameter(torch.tensor(gamma_entity_init))
        self.gamma_style  = nn.Parameter(torch.tensor(gamma_style_init))

        # Whether each branch is active in THIS layer (InstantStyle: injecting
        # image features into all layers degrades high-res structure; restrict
        # to specific layers). Default True for backward compatibility.
        self.style_enabled  = True
        self.entity_enabled = True

        # Back-reference to adapter. Use object.__setattr__ so PyTorch's
        # nn.Module.__setattr__ does NOT register this as a submodule — the
        # adapter holds a reference back to the UNet and would create a
        # circular module tree that breaks set_attn_processor traversal.
        object.__setattr__(self, '_adapter', None)

    # ── Main attention call (replaces AttnProcessor2_0) ──────────────────────

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H_sp, W_sp = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H_sp * W_sp).transpose(1, 2)

        B, S, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, S, B)
            attention_mask = attention_mask.view(B, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        # ── Q (shared across all branches) ───────────────────────────────────
        q = attn.to_q(hidden_states)  # (B, S, inner_dim)

        enc = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attn.norm_cross:
            enc = attn.norm_encoder_hidden_states(enc)

        # ── Text K/V ─────────────────────────────────────────────────────────
        k_text = attn.to_k(enc)
        v_text = attn.to_v(enc)

        inner_dim = k_text.shape[-1]
        n_heads   = attn.heads
        head_dim  = inner_dim // n_heads

        q4 = q.view(B, -1, n_heads, head_dim).transpose(1, 2)         # (B,n,S,d)
        k4 = k_text.view(B, -1, n_heads, head_dim).transpose(1, 2)
        v4 = v_text.view(B, -1, n_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None: q4 = attn.norm_q(q4)
        if attn.norm_k is not None: k4 = attn.norm_k(k4)

        z_text = F.scaled_dot_product_attention(
            q4, k4, v4, attn_mask=attention_mask, dropout_p=0., is_causal=False
        )  # (B, n_heads, S, head_dim)
        z_text = z_text.transpose(1, 2).reshape(B, S, inner_dim)

        # ── Entity branch ─────────────────────────────────────────────────────
        z_entity = None
        if (self.entity_enabled and
                self._adapter is not None and
                self._adapter.entity_tokens is not None and
                self._adapter.entity_bboxes is not None):

            entity_toks = self._adapter.entity_tokens   # (B_cond, n_ent, dim)
                                                         #   or (B_cond, n_ent, K_e, dim)
            entity_bboxes = self._adapter.entity_bboxes

            # Collapse a per-entity token axis (K_e richer patch tokens) into a
            # flat token list, remembering K_e so the bbox bias can be repeated.
            if entity_toks.dim() == 4:
                B_cond, n_ent, K_e, dim = entity_toks.shape
                entity_toks = entity_toks.reshape(B_cond, n_ent * K_e, dim)
            else:
                B_cond, n_ent, dim = entity_toks.shape
                K_e = 1

            # Handle CFG (B = 2 * B_cond: first half uncond, second half cond)
            if B == 2 * B_cond:
                null_ent = torch.zeros_like(entity_toks)
                entity_toks = torch.cat([null_ent, entity_toks], dim=0)
            elif B == B_cond:
                pass  # single-batch (training or non-CFG)

            n_tok = entity_toks.shape[1]   # n_ent * K_e
            # Project in adapter's own dtype (fp32 for training, fp16 for inference)
            # then cast to match Q dtype (hidden_states.dtype) for sdpa.
            proj_dtype = self.entity_to_k.weight.dtype
            k_ent = self.entity_to_k(entity_toks.to(proj_dtype)).to(q4.dtype)
            v_ent = self.entity_to_v(entity_toks.to(proj_dtype)).to(q4.dtype)

            k_ent4 = k_ent.view(B, n_tok, n_heads, head_dim).transpose(1, 2)
            v_ent4 = v_ent.view(B, n_tok, n_heads, head_dim).transpose(1, 2)

            # Spatial resolution for bbox masking
            H_feat = W_feat = int(S ** 0.5)
            attn_bias = make_bbox_attn_bias(
                entity_bboxes, H_feat, W_feat, n_heads, B,
                device=hidden_states.device, dtype=hidden_states.dtype
            )  # (B, n_heads, HW, n_ent)
            if K_e > 1:
                # each entity's K_e tokens share that entity's spatial bbox bias
                # (.contiguous(): SDPA's fused kernels require a contiguous bias)
                attn_bias = attn_bias.repeat_interleave(
                    K_e, dim=-1).contiguous()  # (B,h,HW,n_ent*K_e)

            z_entity = F.scaled_dot_product_attention(
                q4, k_ent4, v_ent4, attn_mask=attn_bias, dropout_p=0., is_causal=False
            )
            z_entity = z_entity.transpose(1, 2).reshape(B, S, inner_dim)

        # ── Style branch ──────────────────────────────────────────────────────
        z_style = None
        if (self.style_enabled and
                self._adapter is not None and
                self._adapter.style_tokens is not None):

            style_toks = self._adapter.style_tokens  # (B_cond, K_g, 768)
            B_cond = style_toks.shape[0]

            if B == 2 * B_cond:
                null_sty = torch.zeros_like(style_toks)
                style_toks = torch.cat([null_sty, style_toks], dim=0)

            K_g = style_toks.shape[1]
            proj_dtype_s = self.style_to_k.weight.dtype
            k_sty = self.style_to_k(style_toks.to(proj_dtype_s)).to(q4.dtype)
            v_sty = self.style_to_v(style_toks.to(proj_dtype_s)).to(q4.dtype)

            k_sty4 = k_sty.view(B, K_g, n_heads, head_dim).transpose(1, 2)
            v_sty4 = v_sty.view(B, K_g, n_heads, head_dim).transpose(1, 2)

            z_style = F.scaled_dot_product_attention(
                q4, k_sty4, v_sty4, dropout_p=0., is_causal=False
            )
            z_style = z_style.transpose(1, 2).reshape(B, S, inner_dim)

        # ── Combine (before to_out) ───────────────────────────────────────────
        combined = z_text
        if z_entity is not None:
            combined = combined + self.gamma_entity * z_entity
        if z_style is not None:
            combined = combined + self.gamma_style * z_style

        combined = combined.to(q.dtype)

        # ── Output projection ─────────────────────────────────────────────────
        combined = attn.to_out[0](combined)
        combined = attn.to_out[1](combined)

        if input_ndim == 4:
            combined = combined.transpose(-1, -2).reshape(B, C, H_sp, W_sp)

        if attn.residual_connection:
            combined = combined + residual
        combined = combined / attn.rescale_output_factor

        return combined


# ── Adapter manager ───────────────────────────────────────────────────────────

class EntityStyleAdapter(nn.Module):
    """
    Injects entity + style cross-attention branches into all attn2 layers of SD UNet.

    After register_to_unet(), call set_conditions() before each generation step
    and clear_conditions() after.  The adapter's parameters are the only
    trainable parts — the UNet remains frozen.

    Trainable params per layer:
      entity_to_k, entity_to_v : (entity_dim × inner_dim) each
      style_to_k,  style_to_v  : (style_dim  × inner_dim) each
      gamma_entity, gamma_style : scalar each

    Total ≈ 16 layers × (2 × 1024 + 2 × 768) × mean_inner_dim
           ≈ 16 × 3584 × 700 ≈ 40M params
    """

    def __init__(
        self,
        unet: nn.Module,
        entity_dim: int = 1024,
        style_dim: int = 768,
        gamma_entity_init: float = 0.1,
        gamma_style_init: float = 0.05,
        style_layers: Optional[list] = None,
        entity_layers: Optional[list] = None,
    ):
        super().__init__()
        self.unet = unet

        # Restrict each branch to specific UNet blocks (InstantStyle).
        # A layer's branch is active iff its (dotted) module name contains one
        # of these substrings. None → all layers (legacy behavior).
        self.style_layers  = style_layers
        self.entity_layers = entity_layers

        # Current per-shot conditions (set by set_conditions)
        self.entity_tokens: Optional[torch.Tensor] = None  # (B_cond, n_ent, entity_dim)
        self.style_tokens:  Optional[torch.Tensor] = None  # (B_cond, K_g, style_dim)
        self.entity_bboxes: Optional[list] = None          # [(x1,y1,x2,y2), ...]

        # Build one processor per attn2 layer
        self.processors = nn.ModuleDict()
        for name, mod in unet.named_modules():
            if name.endswith("attn2"):
                inner_dim = mod.to_q.out_features
                key = name.replace(".", "_")
                proc = EntityStyleAdapterProcessor(
                    inner_dim=inner_dim,
                    entity_dim=entity_dim,
                    style_dim=style_dim,
                    gamma_entity_init=gamma_entity_init,
                    gamma_style_init=gamma_style_init,
                )
                # Bypass nn.Module.__setattr__ to avoid registering adapter
                # as a submodule (would create UNet circular reference).
                object.__setattr__(proc, '_adapter', self)
                # Enable each branch only in the selected blocks
                proc.style_enabled = (
                    True if style_layers is None
                    else any(p in name for p in style_layers))
                proc.entity_enabled = (
                    True if entity_layers is None
                    else any(p in name for p in entity_layers))
                self.processors[key] = proc

        if style_layers is not None:
            n_on = sum(p.style_enabled for p in self.processors.values())
            print(f"[EntityStyleAdapter] style branch active in {n_on}/"
                  f"{len(self.processors)} layers (style_layers={style_layers})")
        if entity_layers is not None:
            n_on = sum(p.entity_enabled for p in self.processors.values())
            print(f"[EntityStyleAdapter] entity branch active in {n_on}/"
                  f"{len(self.processors)} layers (entity_layers={entity_layers})")

    def register_to_unet(self):
        """Replace all attn2 processors with EntityStyleAdapterProcessor."""
        new_procs = {}
        for name, mod in self.unet.named_modules():
            full_proc_name = f"{name}.processor"
            if full_proc_name not in self.unet.attn_processors:
                continue
            if name.endswith("attn2"):
                key = name.replace(".", "_")
                new_procs[full_proc_name] = self.processors[key]
            else:
                # Keep existing processor for attn1 (GLIGEN gated self-attention)
                new_procs[full_proc_name] = self.unet.attn_processors[full_proc_name]
        self.unet.set_attn_processor(new_procs)

    def set_conditions(
        self,
        entity_tokens: torch.Tensor,
        entity_bboxes: list,
        style_tokens: Optional[torch.Tensor] = None,
    ):
        """
        Set per-shot conditions before each UNet forward pass.

        entity_tokens : (1, n_ent, entity_dim)   CLIP embeddings of entity refs
        entity_bboxes : [(x1,y1,x2,y2), ...]     normalized [0,1] coords, len=n_ent
        style_tokens  : (1, K_g, style_dim) | None
        """
        self.entity_tokens = entity_tokens
        self.entity_bboxes = entity_bboxes
        self.style_tokens  = style_tokens

    def clear_conditions(self):
        self.entity_tokens = None
        self.entity_bboxes = None
        self.style_tokens  = None

    def trainable_parameters(self) -> list:
        return list(self.processors.parameters())

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.processors.parameters())
        entity_kv = sum(
            p.numel() for proc in self.processors.values()
            for name, p in proc.named_parameters()
            if "entity" in name and ("to_k" in name or "to_v" in name)
        )
        style_kv = sum(
            p.numel() for proc in self.processors.values()
            for name, p in proc.named_parameters()
            if "style" in name and ("to_k" in name or "to_v" in name)
        )
        return {"total": total, "entity_kv": entity_kv, "style_kv": style_kv}

    def save(self, path: str, epoch: int = 0, loss: float = 0.):
        torch.save({
            "epoch":  epoch,
            "loss":   loss,
            "model":  self.processors.state_dict(),
            "config": {
                "entity_dim": self.processors[
                    next(iter(self.processors))
                ].entity_to_k.in_features,
                "style_dim": self.processors[
                    next(iter(self.processors))
                ].style_to_k.in_features,
                "style_layers": self.style_layers,
                "entity_layers": self.entity_layers,
            },
        }, path)

    @classmethod
    def load(cls, path: str, unet: nn.Module, device: str = "cuda") -> "EntityStyleAdapter":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        adapter = cls(
            unet,
            entity_dim=cfg.get("entity_dim", 1024),
            style_dim=cfg.get("style_dim", 768),
            style_layers=cfg.get("style_layers", None),
            entity_layers=cfg.get("entity_layers", None),
        )
        adapter.processors.load_state_dict(ckpt["model"])
        adapter.processors.to(device)
        return adapter
