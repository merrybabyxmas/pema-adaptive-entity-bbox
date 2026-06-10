"""
LISA-V Phase 3: Attention Injection

Custom self-attention processor that applies tube masks to prevent entity
feature bleeding during diffusion, plus IP-Adapter regional cross-attention
control via diffusers' built-in ip_adapter_masks mechanism.

Two levels of control:
  1. IP-Adapter Regional Control (built-in) -- handled by diffusers via
     ``cross_attention_kwargs["ip_adapter_masks"]``
  2. Self-Attention Masking (custom) -- tokens outside an entity's tube
     are penalized when attending to tokens inside it:
     Softmax(QK^T / sqrt(d) - lambda * (1 - M_tube)) V
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

def resize_mask_for_attention(mask, res):
    """Resize a [H,W] (or [F,H,W]) mask to target (h,w) via bilinear interp."""
    import torch as _t
    h, w = res
    m = mask.float()
    if m.ndim == 2:
        out = _t.nn.functional.interpolate(m[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    else:  # [F,H,W]
        out = _t.nn.functional.interpolate(m[:, None], size=(h, w), mode="bilinear", align_corners=False)[:, 0]
    return out

logger = logging.getLogger(__name__)


class LISASelfAttentionProcessor:
    """Custom self-attention processor that wraps an existing processor.

    Applies LISA-V's isolation logic to self-attention layers while
    delegating cross-attention to the original processor (preserving IP-Adapter).

    Supports two modes:
      1. Penalty mode (legacy): Large negative penalty on cross-region attention scores
      2. RAAN mode (Region-Aware Attention Normalization): Independent softmax per region
         with soft blending ratio for smooth entity boundaries

    Supports both 2D image masks [H, W] and 3D video mask tubes [F, H, W].
    For 3D masks (AnimateDiff), each frame in the batch gets its own
    per-frame spatial region mask — AnimateDiff collapses frames into the
    batch dimension for spatial attention (batch_size = B*F, seq_len = H*W).
    """

    def __init__(
        self,
        original_processor: object,
        entity_masks: Dict[str, torch.Tensor],
        temporal_mask_lambda: float = 1e6,
        use_raan: bool = False,
        raan_blend_ratio: float = 0.8,
    ) -> None:
        """
        Args:
            original_processor: The original processor (e.g. AttnProcessor2_0 or IPAdapterAttnProcessor).
            entity_masks: Dict mapping entity_id to [H, W] or [F, H, W] masks.
            temporal_mask_lambda: Penalty applied to cross-region attention (penalty mode only).
            use_raan: If True, use Region-Aware Attention Normalization instead of penalty.
            raan_blend_ratio: Blending ratio for RAAN (0=global softmax, 1=pure region softmax).
        """
        self.original_processor = original_processor
        self.use_raan = use_raan
        self.raan_blend_ratio = raan_blend_ratio
        self._call_count = 0
        self._raan_applied_count = 0
        self._penalty_applied_count = 0

        # Clamping
        if temporal_mask_lambda > 1e4:
            self.temporal_mask_lambda = 1e4
        else:
            self.temporal_mask_lambda = temporal_mask_lambda

        sorted_eids = sorted(entity_masks.keys())
        first_mask = entity_masks[sorted_eids[0]]
        self.is_3d = first_mask.ndim == 3  # [F, H, W] video mask tube

        if self.is_3d:
            num_frames = first_mask.shape[0]
            self.spatial_len = first_mask.shape[1] * first_mask.shape[2]  # H*W

            if self.use_raan:
                # RAAN: precompute per-frame region assignments [F, N+1, H*W]
                self.per_frame_regions = []
                for f in range(num_frames):
                    frame_masks = []
                    for eid in sorted_eids:
                        frame_masks.append(entity_masks[eid][f].reshape(-1))  # [H*W]
                    stacked = torch.stack(frame_masks, dim=0)  # [N, H*W]
                    bg = (1.0 - stacked.sum(dim=0)).clamp(0, 1)
                    regions_f = torch.cat([stacked, bg.unsqueeze(0)], dim=0)  # [N+1, H*W]
                    self.per_frame_regions.append(regions_f)
                self.per_frame_penalties = None
            else:
                # Penalty mode: precompute per-frame penalty matrices
                self.per_frame_regions = None
                self.per_frame_penalties = []
                for f in range(num_frames):
                    frame_masks = []
                    for eid in sorted_eids:
                        frame_masks.append(entity_masks[eid][f].reshape(-1))  # [H*W]
                    stacked = torch.stack(frame_masks, dim=0)  # [N, H*W]
                    bg = (1.0 - stacked.sum(dim=0)).clamp(0, 1)
                    regions_f = torch.cat([stacked, bg.unsqueeze(0)], dim=0)  # [N+1, H*W]
                    region_sim = torch.matmul(regions_f.transpose(0, 1), regions_f)
                    penalty = self.temporal_mask_lambda * (1.0 - region_sim)
                    self.per_frame_penalties.append(penalty)
            self.regions = None
        else:
            # 2D path
            self.per_frame_penalties = None
            self.per_frame_regions = None
            masks_flat = []
            for eid in sorted_eids:
                masks_flat.append(entity_masks[eid].reshape(-1))

            if masks_flat:
                stacked = torch.stack(masks_flat, dim=0)  # [N, L]
                bg_mask = (1.0 - stacked.sum(dim=0)).clamp(0, 1)
                self.regions = torch.cat([stacked, bg_mask.unsqueeze(0)], dim=0)
            else:
                self.regions = None

    def _raan_softmax(
        self,
        attn_scores: torch.Tensor,
        regions: torch.Tensor,
    ) -> torch.Tensor:
        """Region-Aware Attention Normalization: independent softmax per region.

        Each query token attends to key tokens with a softmax denominator computed
        only within the query's dominant region. This ensures each region's attention
        sum = 1.0 independently, preventing attention energy dilution as entity count grows.

        Uses EXCLUSIVE region assignments via argmax: each token belongs to exactly
        one region.  Query tokens in region r can only attend to key tokens also
        assigned to region r (plus a global-softmax blend for coherence).

        The final output is a blend of region-local softmax and global softmax:
            attn = λ * region_softmax + (1-λ) * global_softmax

        Args:
            attn_scores: [heads, H*W, H*W] raw QK^T scores.
            regions: [N+1, H*W] soft region membership (entities + background).

        Returns:
            [heads, H*W, H*W] attention probabilities.
        """
        device = attn_scores.device
        dtype = attn_scores.dtype
        regions = regions.to(device=device, dtype=dtype)
        num_regions, seq_len = regions.shape

        # 1. Global softmax (baseline)
        global_probs = F.softmax(attn_scores, dim=-1)  # [heads, L, L]

        # 2. Region-local softmax with EXCLUSIVE assignments
        # Each token is assigned to exactly one region via argmax
        token_region = regions.argmax(dim=0)  # [L]

        # Build a single key-suppression mask: for each region, keys outside
        # that region get -1e4 penalty.  We construct the per-query penalty
        # by broadcasting from the query's region assignment.
        # key_allowed[q, k] = True iff token_region[q] == token_region[k]
        # Instead of looping per region, use vectorized approach:
        region_probs = torch.zeros_like(attn_scores)  # [heads, L, L]

        for r in range(num_regions):
            query_mask = (token_region == r)  # [L] bool — queries in region r
            if not query_mask.any():
                continue

            key_mask = (token_region == r)  # [L] bool — keys in region r (exclusive!)

            # For queries in this region, suppress all keys outside this region
            # Use in-place operations on a view to avoid cloning full matrix
            region_scores = attn_scores[:, query_mask, :]  # [heads, n_q, L]
            # Apply penalty to keys outside this region
            penalty = torch.zeros(seq_len, device=device, dtype=dtype)
            penalty[~key_mask] = 1e4
            region_scores = region_scores - penalty.unsqueeze(0).unsqueeze(0)

            region_softmax = F.softmax(region_scores, dim=-1)  # [heads, n_q, L]
            region_probs[:, query_mask, :] = region_softmax

        # 3. Blend region-local and global softmax
        blended = self.raan_blend_ratio * region_probs + (1.0 - self.raan_blend_ratio) * global_probs

        return blended

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process attention with region-based isolation for self-attention."""
        # Cross-attention: delegate to the original processor (very important for IP-Adapter!)
        if encoder_hidden_states is not None:
            return self.original_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kwargs
            )

        # Self-attention: apply LISA masking logic
        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, seq_len, _ = hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scale = head_dim ** -0.5
        attn_scores = torch.matmul(query, key.transpose(-1, -2)) * scale

        self._call_count += 1

        if self.use_raan:
            # RAAN mode: region-wise independent softmax
            if self.is_3d and self.per_frame_regions is not None and seq_len == self.spatial_len:
                num_frames = len(self.per_frame_regions)
                attn_probs = torch.zeros_like(attn_scores)
                self._raan_applied_count += 1
                for b in range(batch_size):
                    f_idx = b % num_frames
                    regions = self.per_frame_regions[f_idx]
                    attn_probs[b] = self._raan_softmax(attn_scores[b], regions)
            elif not self.is_3d and self.regions is not None and self.regions.shape[1] == seq_len:
                self._raan_applied_count += 1
                attn_probs = torch.zeros_like(attn_scores)
                for b in range(batch_size):
                    attn_probs[b] = self._raan_softmax(attn_scores[b], self.regions)
            else:
                attn_probs = attn_scores.softmax(dim=-1)
        else:
            # Penalty mode (legacy)
            if self.is_3d and self.per_frame_penalties is not None and seq_len == self.spatial_len:
                num_frames = len(self.per_frame_penalties)
                for b in range(batch_size):
                    f_idx = b % num_frames
                    penalty = self.per_frame_penalties[f_idx].to(
                        device=attn_scores.device, dtype=attn_scores.dtype
                    )
                    attn_scores[b] = attn_scores[b] - penalty.unsqueeze(0)
            elif not self.is_3d and self.regions is not None and self.regions.shape[1] == seq_len:
                regions = self.regions.to(device=attn_scores.device, dtype=attn_scores.dtype)
                region_sim = torch.matmul(regions.transpose(0, 1), regions)
                penalty = self.temporal_mask_lambda * (1.0 - region_sim)
                attn_scores = attn_scores - penalty.unsqueeze(0).unsqueeze(0)

            attn_probs = attn_scores.softmax(dim=-1)

        attn_probs = attn_probs.to(value.dtype)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, seq_len, inner_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Injection helpers
# ---------------------------------------------------------------------------

def _get_attention_layer_resolution(
    name: str,
    unet,
    generation_resolution: Optional[Tuple[int, int]] = None,
) -> Optional[Tuple[int, int]]:
    """Infer the spatial resolution for a named attention layer.

    SDXL UNet attention layers operate at multiple resolutions depending
    on the block.  Uses the actual generation resolution (pixel space) to
    compute correct latent-space attention resolutions.

    Args:
        name: Dotted module path.
        unet: The UNet model (unused, kept for API compat).
        generation_resolution: (H, W) in pixel space. Required for
            non-square generation (e.g. 512x768).

    Returns (H, W) or None if unable to determine.
    """
    if generation_resolution is not None:
        # VAE downsamples by 8x
        base_h = generation_resolution[0] // 8
        base_w = generation_resolution[1] // 8
    else:
        # Fallback: assume 1024x1024 → 128x128 latent
        base_h, base_w = 128, 128

    # SDXL UNet downsample factors by block:
    #  down_blocks.0 -> 1x (no downsample)
    #  down_blocks.1 -> 2x
    #  down_blocks.2 -> 4x
    #  mid_block     -> 4x (same as down_blocks.2)
    #  up_blocks.0   -> 4x
    #  up_blocks.1   -> 2x
    #  up_blocks.2   -> 1x
    downsample_map = {
        "down_blocks.0": 1,
        "down_blocks.1": 2,
        "down_blocks.2": 4,
        "mid_block": 4,
        "up_blocks.0": 4,
        "up_blocks.1": 2,
        "up_blocks.2": 1,
    }
    for prefix, factor in downsample_map.items():
        if name.startswith(prefix):
            return (base_h // factor, base_w // factor)
    return None


def _should_inject(name: str, injection_layers: str) -> bool:
    """Check if the layer identified by *name* should receive injection.

    Args:
        name: Dotted module path (e.g. "up_blocks.1.attentions.0.transformer_blocks.0.attn1").
        injection_layers: One of "all", "up_only", "mid_up".
    """
    # Skip 128x128 resolution layers (down_blocks.0, up_blocks.2) to avoid OOM
    # on the [16384, 16384] penalty matrix. Entity isolation is structural,
    # so 32x32/64x64 layers are sufficient.
    if name.startswith("down_blocks.0") or name.startswith("up_blocks.2"):
        return False
    if injection_layers == "all":
        return True
    if injection_layers == "up_only":
        return name.startswith("up_blocks")
    if injection_layers == "mid_up":
        return name.startswith("up_blocks") or name.startswith("mid_block")
    return True


def inject_attention_control(
    pipe,
    entity_masks: Dict[str, torch.Tensor],
    config: Optional[Dict] = None,
    generation_resolution: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    """Inject LISA self-attention processors into the UNet.

    Args:
        pipe: A StableDiffusionXLPipeline (with IP-Adapter loaded).
        entity_masks: Dict[entity_id, Tensor[H, W]] entity masks at the
            original generation resolution.
        config: dict with keys ``temporal_mask_lambda``, ``injection_layers``.
        generation_resolution: (H, W) pixel-space resolution of generation.
            Required for correct attention layer resolution mapping.

    Returns:
        The original attention processors dict so you can restore them later.
    """
    config = config or {}
    temporal_mask_lambda = config.get("temporal_mask_lambda", 1e6)
    injection_layers = config.get("injection_layers", "all")
    use_raan = config.get("use_raan", False)
    raan_blend_ratio = config.get("raan_blend_ratio", 0.8)

    # Compute union (combined) tube mask at original resolution
    mask_list = list(entity_masks.values())
    if mask_list:
        combined = torch.stack(mask_list, dim=0).max(dim=0).values  # [H, W]
    else:
        logger.warning("No entity masks provided; skipping attention injection")
        return pipe.unet.attn_processors

    # Save originals for later restoration
    original_processors = dict(pipe.unet.attn_processors)

    new_processors = {}
    injected_count = 0

    for name, proc in pipe.unet.attn_processors.items():
        # Only inject into spatial self-attention layers (attn1), NOT motion_modules
        # Motion module temporal attention operates across frames (seq_len = F)
        # and should not be masked — it provides the motion prior.
        is_self_attn = ".attn1." in name or name.endswith(".attn1.processor")
        is_motion_module = "motion_module" in name
        if is_self_attn and not is_motion_module and _should_inject(name, injection_layers):
            res = _get_attention_layer_resolution(name, pipe.unet, generation_resolution)
            if res is not None:
                # Resize all entity masks to this layer's resolution
                resized_masks = {}
                for eid, m in entity_masks.items():
                    resized_masks[eid] = resize_mask_for_attention(m, res)
                
                new_processors[name] = LISASelfAttentionProcessor(
                    original_processor=proc,
                    entity_masks=resized_masks,
                    temporal_mask_lambda=temporal_mask_lambda,
                    use_raan=use_raan,
                    raan_blend_ratio=raan_blend_ratio,
                )
                injected_count += 1
            else:
                # Can't determine resolution; keep original
                new_processors[name] = proc
        else:
            # Cross-attention or non-target layer; keep original
            new_processors[name] = proc

    pipe.unet.set_attn_processor(new_processors)
    mode_str = f"RAAN(blend={raan_blend_ratio})" if use_raan else f"Penalty(lambda={temporal_mask_lambda:.0e})"
    logger.info(
        "Injected LISA self-attention processors into %d layers (policy=%s, mode=%s)",
        injected_count, injection_layers, mode_str,
    )

    return original_processors


def restore_attention_processors(
    pipe,
    original_processors: Optional[Dict[str, object]] = None,
) -> None:
    """Restore original attention processors after generation.

    If *original_processors* is provided, set them directly. Otherwise
    fall back to the default ``AttnProcessor2_0``.
    """
    if original_processors is not None:
        # Log diagnostic stats from LISA processors before restoring
        for name, proc in pipe.unet.attn_processors.items():
            if isinstance(proc, LISASelfAttentionProcessor) and proc._call_count > 0:
                mode = "RAAN" if proc.use_raan else "Penalty"
                applied = proc._raan_applied_count if proc.use_raan else proc._penalty_applied_count
                logger.info(
                    "  [%s] %s: calls=%d, masking_applied=%d (spatial_len=%s)",
                    mode, name.split(".")[-3] if "." in name else name,
                    proc._call_count, applied,
                    getattr(proc, 'spatial_len', 'N/A'),
                )
                break  # Just log one representative layer
        pipe.unet.set_attn_processor(original_processors)
        logger.debug("Restored original attention processors")
    else:
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            pipe.unet.set_attn_processor(AttnProcessor2_0())
        except ImportError:
            logger.warning("AttnProcessor2_0 not available; processors left as-is")
    logger.info("Attention processors restored")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== attention_injection self-test ===")

    # Test processor instantiation with a dummy mask
    mask = torch.rand(64, 64)
    proc = LISASelfAttentionProcessor(
        combined_tube_mask=mask,
        temporal_mask_lambda=1e6,
    )
    logger.info("Processor created, tube_mask_flat shape: %s", proc.tube_mask_flat.shape)
    assert proc.tube_mask_flat.shape == (64 * 64,)

    # Test resolution map helper
    assert _get_attention_layer_resolution("down_blocks.0.something", None) == (128, 128)
    assert _get_attention_layer_resolution("mid_block.something", None) == (32, 32)
    assert _get_attention_layer_resolution("up_blocks.2.something", None) == (128, 128)

    # Test injection policy
    assert _should_inject("up_blocks.1.attn1", "all") is True
    assert _should_inject("down_blocks.0.attn1", "up_only") is False
    assert _should_inject("up_blocks.1.attn1", "up_only") is True
    assert _should_inject("mid_block.attn1", "mid_up") is True

    logger.info("=== attention_injection self-test PASSED ===")
