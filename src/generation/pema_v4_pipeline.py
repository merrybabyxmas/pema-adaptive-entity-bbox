"""
PEMA v4: Phase 4 generation pipeline.

Combines:
  - GLIGEN for text + spatial layout (phase 3 baseline)
  - EntityStyleAdapter for entity identity cross-attention (phase 4 NEW)
  - StyleEncoder for style cross-attention (phase 4 NEW)

Attention structure per denoising step:
  z_text   = cross_attn(Q, K_text, V_text)         [GLIGEN text + grounding]
  z_entity = cross_attn(Q, K_ent, V_ent, bbox_mask) [entity memory, localized]
  z_style  = cross_attn(Q, K_style, V_style)        [global style]
  z = (z_text + γ_e * z_entity + γ_g * z_style) → to_out

Phase 4 changes vs Phase 3:
  - Entity memory fed via dedicated cross-attention branch (not GLIGEN grounding)
  - Style fed via dedicated cross-attention branch (not GLIGEN grounding)
  - GLIGEN grounding still active for layout positioning (entity name phrases)
  - Both entity and style branches are bbox-/globally-aware at the attention level
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
from typing import Optional

from src.memory.memory_bank import EntityMemoryBank
from src.model.style_encoder import StyleEncoder
from src.model.entity_style_adapter import EntityStyleAdapter
from src.generation.layout_adapter import plan_to_layout, draw_layout_on_image
from src.generation.pema_pipeline import (
    deoverlap_boxes, bootstrap_entity_image,
    GLIGEN_MODEL, _crop_bbox,
)
from src.generation.pema_v2_pipeline import (
    _load_memory_gligen, extract_style_background,
)
from src.generation.memory_gligen_pipeline import MemoryGLIGENPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Phase 4 generation function ───────────────────────────────────────────────

def generate_with_pema_v4(
    plan_output: dict,
    memory_bank: EntityMemoryBank,
    output_dir: str,
    adapter: EntityStyleAdapter,
    gligen_pipe: Optional[MemoryGLIGENPipeline] = None,
    style_encoder: Optional[StyleEncoder] = None,
    entity_encoder=None,           # EntityEncoder
    device: str = "cuda",
    update_memory: bool = True,
    gligen_steps: int = 30,
    guidance_scale: float = 7.5,
    n_style_tokens: int = 4,
    use_phase3_grounding: bool = True,  # keep GLIGEN grounding for layout
    style_ref_tokens: Optional[torch.Tensor] = None,  # (K_g,768) global style
) -> list[dict]:
    """
    PEMA v4: EntityStyleAdapter + (optional) GLIGEN layout grounding.

    Entity memory → entity cross-attention branch (bbox-localized)
    Style reference → style cross-attention branch (global)

    use_phase3_grounding=True:  also inject entity name + memory as GLIGEN phrases
                                (redundant but safe — doubles identity signal)
    use_phase3_grounding=False: entity identity via cross-attn adapter only
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layouts = plan_to_layout(plan_output)

    if gligen_pipe is None:
        gligen_pipe = _load_memory_gligen(GLIGEN_MODEL, device)

    enc_dtype = gligen_pipe.text_encoder.dtype

    # Global style token. If a dedicated style reference is provided, use it for
    # ALL shots (incl. shot 0) — uniform video-level style. Otherwise fall back
    # to bootstrapping from shot 0's background (legacy: shots 1+ only).
    style_tokens_global: Optional[torch.Tensor] = None
    if style_ref_tokens is not None:
        style_tokens_global = style_ref_tokens
        logger.info("Global style token from dedicated style reference "
                    "(applied to all shots)")

    results = []
    for layout in layouts:
        shot_id  = layout["shot_id"]
        prompt   = layout["prompt"]
        entities = layout["entities"]
        active   = [e for e in entities if memory_bank.has(e["name"])]

        # ── Deoverlap bboxes ──────────────────────────────────────────────
        raw_boxes  = {e["name"]: e["box_xyxy"] for e in entities}
        deov_boxes = deoverlap_boxes(raw_boxes)
        for e in entities:
            e["box_xyxy"] = deov_boxes[e["name"]]

        # ── Global style token (applied regardless of entity presence) ───
        sty_toks = None
        if style_tokens_global is not None:
            sty_toks = style_tokens_global.unsqueeze(0).to(
                device=device, dtype=enc_dtype
            )  # (1, K_g, 768)

        # ── Phase 4: set adapter conditions (entity cross-attention) ──────
        if active:
            ent_clips = []
            ent_bboxes = []
            for e in active:
                clip_emb = memory_bank.retrieve_tokens(e["name"]).to(device)  # (1024,)
                ent_clips.append(clip_emb)
                ent_bboxes.append(e["box_xyxy"])

            entity_tokens = torch.stack(ent_clips).unsqueeze(0).to(
                device=device, dtype=enc_dtype
            )  # (1, n_ent, 1024)

            adapter.set_conditions(entity_tokens, ent_bboxes, sty_toks)
        else:
            # no entities this shot, but still apply global style
            adapter.set_conditions(None, [], sty_toks)

        # ── Phase 3 GLIGEN grounding (layout + optional entity memory) ────
        if use_phase3_grounding:
            gligen_phrases = [e["name"] for e in entities]
            gligen_boxes   = [e["box_xyxy"] for e in entities]
            gligen_phrase_embs = None   # use GLIGEN's own tokenization
        else:
            # No GLIGEN grounding (entity identity from adapter only)
            gligen_phrases = [prompt]
            gligen_boxes   = [[0., 0., 1., 1.]]
            gligen_phrase_embs = None

        logger.info(
            f"Shot {shot_id} | entities={[e['name'] for e in active]} "
            f"phase4_adapter={'active' if active else 'none'} "
            f"style={'active' if sty_toks is not None else 'none'}"
        )

        # ── Generate ──────────────────────────────────────────────────────
        gen_result = gligen_pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, cartoon, duplicate, text",
            gligen_phrases=gligen_phrases,
            gligen_boxes=gligen_boxes,
            gligen_phrase_embeddings=gligen_phrase_embs,
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=gligen_steps,
            guidance_scale=guidance_scale,
            height=512, width=512,
        )
        gen_img = gen_result.images[0]

        # Clear adapter conditions after generation
        adapter.clear_conditions()

        # ── Extract style from shot 0 background (only if no dedicated ref) ──
        if (style_ref_tokens is None and shot_id == 0
                and style_encoder is not None and entity_encoder is not None):
            all_bboxes = [e["box_xyxy"] for e in entities]
            bg_img     = extract_style_background(gen_img, all_bboxes)
            bg_path    = out_path / "shot_000_style_bg.png"
            bg_img.save(str(bg_path))
            with torch.no_grad():
                bg_clip = entity_encoder.encode(bg_img).to(device)
                style_tokens_global = style_encoder(
                    bg_clip.unsqueeze(0)
                ).squeeze(0)  # (K_g, 768)
            logger.info(
                f"  Style tokens extracted from shot 0 bg "
                f"(K_g={style_tokens_global.shape[0]})"
            )

        # ── Save ──────────────────────────────────────────────────────────
        img_bbox  = draw_layout_on_image(gen_img.copy(), layout)
        shot_path = out_path / f"shot_{shot_id:03d}.png"
        bbox_path = out_path / f"shot_{shot_id:03d}_bbox.png"
        gen_img.save(str(shot_path))
        img_bbox.save(str(bbox_path))

        # ── Memory update ─────────────────────────────────────────────────
        update_log = {}
        if update_memory:
            for e in active:
                crop, _ = _crop_bbox(gen_img, e["box_xyxy"])
                crop_path = out_path / f"shot_{shot_id:03d}_crop_{e['name']}.png"
                crop.save(str(crop_path))
                accepted = memory_bank.update(e["name"], crop)
                update_log[e["name"]] = "updated" if accepted else "rejected"

        results.append({
            "shot_id":        shot_id,
            "path":           str(shot_path),
            "bbox_path":      str(bbox_path),
            "active_entities":[e["name"] for e in active],
            "memory_updates": update_log,
            "mode":           "phase4-adapter",
        })
        logger.info(
            f"  shot {shot_id} done | memory: {update_log}"
        )

    return results
