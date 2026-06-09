"""
PEMA v2: Presence-Aware Entity Memory Diffusion

Phase 3 architecture: text + entity memory + global style tokens
all condition diffusion from t=T via GLIGEN's bbox-localized gated attention.

Three conditioning branches per denoising step:
  z = z_text + γ_e * z_entity + γ_g * z_style

  z_text:   GLIGEN prompt cross-attention (global, full image)
  z_entity: entity memory grounding (bbox-localized per entity)
  z_style:  style token grounding     (full-image bbox, γ_g=0.1~0.3)

Entity grounding formula (Phase 1):
  g_e = text_enc(name) + λ_m * r_e
  r_e = orthogonalize(ResidualEntityConditioner(M_e), text_enc(name))
  r_e ⊥ text_enc(name)  →  r_e carries identity not in text

Style grounding formula (Phase 2):
  G_style = StyleEncoder(CLIP_img(background))  →  (K_g, 768)
  Extracted from shot 0 background (entity regions masked).
  Injected as K_g extra grounding phrases with bbox=[0,0,1,1].
  Applied from shot 1 onward (shot 0 provides the anchor).

Phase 0: GlobalStyle pixel blend disabled (use_style_blend=False).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

from src.memory.memory_bank import EntityMemoryBank
from src.model.entity_memory_conditioner import EntityMemoryConditioner
from src.model.residual_entity_conditioner import ResidualEntityConditioner
from src.model.style_encoder import StyleEncoder
from src.generation.layout_adapter import plan_to_layout, draw_layout_on_image
from src.generation.pema_pipeline import (
    deoverlap_boxes, bootstrap_entity_image,
    GlobalStyleMemory, _load_gligen,
    GLIGEN_MODEL, SD15_MODEL,
)
from src.generation.memory_gligen_pipeline import MemoryGLIGENPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Pipeline loader ───────────────────────────────────────────────────────────

def _load_memory_gligen(model_id: str, device: str) -> MemoryGLIGENPipeline:
    pipe = MemoryGLIGENPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


# ── Phase 2: Style background extraction ─────────────────────────────────────

def extract_style_background(
    image: Image.Image,
    entity_bboxes: list[list[float]],
    blur_radius: int = 40,
) -> Image.Image:
    """
    Replace entity bbox regions with heavy blur to expose background style.
    Used to extract style tokens that don't carry entity-specific content.

    entity_bboxes: [[x1,y1,x2,y2], ...] in normalized [0,1] coords.
    """
    W, H = image.size
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for box in entity_bboxes:
        pad = int(0.05 * min(W, H))
        x1 = max(0, int(box[0] * W) - pad)
        y1 = max(0, int(box[1] * H) - pad)
        x2 = min(W, int(box[2] * W) + pad)
        y2 = min(H, int(box[3] * H) + pad)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    return Image.composite(blurred, image, mask)


# ── Phase 3: Grounding embedding builder ─────────────────────────────────────

def _build_grounding_embeddings(
    entities: list[dict],
    memory_bank: EntityMemoryBank,
    conditioner,           # ResidualEntityConditioner | EntityMemoryConditioner | None
    text_encoder,
    tokenizer,
    device: str,
    dtype,
    text_weight: float = 1.0,
    memory_weight: float = 1.0,
    style_tokens: torch.Tensor | None = None,  # (K_g, 768) global style tokens
    style_weight: float = 0.2,                  # γ_style
) -> tuple[torch.Tensor, list[str], list[list[float]]]:
    """
    Build grounding embeddings for entity + style conditioning.

    Returns:
        phrase_embeds:  (n_entities + K_g, 768)  — all grounding embeddings
        gligen_phrases: list of phrase strings    — entity names + "style_k"
        gligen_boxes:   list of bboxes            — entity bboxes + [[0,0,1,1]] * K_g

    Entity branch (Phase 1 — residual conditioner):
        r_e = ResidualEntityConditioner(M_e)   # unit norm
        r_e ← orthogonalize(r_e, text_enc(e)) # remove text direction
        r_e ← normalize(r_e)                   # unit norm after orth
        g_e = text_weight * text_enc(e) + memory_weight * r_e

    Style branch (Phase 2):
        g_style_k = style_tokens[k] * style_weight
        bbox = [0, 0, 1, 1]  (full image, no spatial restriction)
    """
    is_residual = isinstance(conditioner, ResidualEntityConditioner)
    phrase_embeds = []
    gligen_phrases = []
    gligen_boxes   = []

    # ── Entity branch ─────────────────────────────────────────────────────
    for e in entities:
        name = e["name"]
        tok  = tokenizer(name, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_emb = text_encoder(**tok).pooler_output.float()    # (1, 768)

        if conditioner is not None and memory_bank.has(name):
            clip_emb = memory_bank.retrieve_tokens(name).unsqueeze(0).to(device)  # (1, 1024)
            with torch.no_grad():
                mem_out = conditioner(clip_emb).float()             # (1, 768)

            if is_residual:
                t_n     = F.normalize(text_emb, dim=-1)
                mem_out = mem_out - (mem_out * t_n).sum(-1, keepdim=True) * t_n
                mem_out = F.normalize(mem_out, dim=-1)

            grounding = text_weight * text_emb + memory_weight * mem_out
        else:
            grounding = text_weight * text_emb

        phrase_embeds.append(grounding.squeeze(0).to(dtype))
        gligen_phrases.append(name)
        gligen_boxes.append(e["box_xyxy"])

    # ── Style branch (Phase 2) ────────────────────────────────────────────
    if style_tokens is not None:
        K_g = style_tokens.shape[0]
        for k in range(K_g):
            token = style_tokens[k].to(device=device, dtype=dtype) * style_weight
            phrase_embeds.append(token)
            gligen_phrases.append("global_style")
            gligen_boxes.append([0., 0., 1., 1.])
        logger.debug(f"  Style branch: K_g={K_g}, γ_style={style_weight:.2f}")

    return torch.stack(phrase_embeds), gligen_phrases, gligen_boxes


# ── Phase 3: Main generation function ────────────────────────────────────────

def generate_with_pema_v2(
    plan_output: dict,
    memory_bank: EntityMemoryBank,
    output_dir: str,
    gligen_pipe: MemoryGLIGENPipeline | None = None,
    conditioner=None,                   # ResidualEntityConditioner (preferred)
    style_encoder: StyleEncoder | None = None,  # Phase 2: global style tokens
    entity_encoder=None,                # EntityEncoder (needed for style encoding)
    style_memory: GlobalStyleMemory | None = None,
    device: str = "cuda",
    update_memory: bool = True,
    gligen_steps: int = 30,
    text_weight: float = 1.0,
    memory_weight: float = 1.0,
    style_weight: float = 0.2,          # γ_style — scales style token magnitude
    use_style_blend: bool = False,       # ablation only, off by default
) -> list[dict]:
    """
    PEMA v2 Phase 3: entity memory + global style tokens condition denoising from t=T.

    Conditioning per shot:
      text_tokens   (full image, from prompt)
      entity_tokens (bbox-localized, from ResidualEntityConditioner)
      style_tokens  (full image, from StyleEncoder, γ_style=0.2)

    Style tokens extracted from shot 0 background (entity regions masked).
    Applied from shot 1 onward. Shot 0: entity+text only.

    conditioner=None        → text-only (GLIGEN baseline)
    style_encoder=None      → no style conditioning (entity+text only)
    both provided           → full Phase 3 method
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layouts = plan_to_layout(plan_output)

    if gligen_pipe is None:
        gligen_pipe = _load_memory_gligen(GLIGEN_MODEL, device)
        logger.info(f"Loaded MemoryGLIGENPipeline: {GLIGEN_MODEL}")

    if use_style_blend and style_memory is None:
        style_memory = GlobalStyleMemory()

    conditioner_type = (
        "residual"                 if isinstance(conditioner, ResidualEntityConditioner)
        else "text-proj-ablation"  if conditioner is not None
        else "text-only"
    )
    has_style = style_encoder is not None and entity_encoder is not None
    mode_full = (
        f"{conditioner_type}+style" if has_style
        else conditioner_type
    )
    enc_dtype = gligen_pipe.text_encoder.dtype

    # Style tokens: initialized from shot 0 background, used for shots 1+
    style_tokens: torch.Tensor | None = None

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
            orig  = e["box_xyxy"]
            fixed = deov_boxes[e["name"]]
            if fixed != orig:
                logger.info(
                    f"  deoverlap {e['name']}: "
                    f"[{orig[0]:.2f},{orig[1]:.2f},{orig[2]:.2f},{orig[3]:.2f}] → "
                    f"[{fixed[0]:.2f},{fixed[1]:.2f},{fixed[2]:.2f},{fixed[3]:.2f}]"
                )
            e["box_xyxy"] = fixed

        # ── Build grounding embeddings (entity + style) ───────────────────
        phrase_embeds, gligen_phrases, gligen_boxes = _build_grounding_embeddings(
            entities=entities,
            memory_bank=memory_bank,
            conditioner=conditioner,
            text_encoder=gligen_pipe.text_encoder,
            tokenizer=gligen_pipe.tokenizer,
            device=device,
            dtype=enc_dtype,
            text_weight=text_weight,
            memory_weight=memory_weight,
            style_tokens=style_tokens,   # None for shot 0
            style_weight=style_weight,
        )

        n_style = 0 if style_tokens is None else style_tokens.shape[0]
        logger.info(
            f"Shot {shot_id} | [{conditioner_type}] "
            f"entities={[e['name'] for e in entities]} "
            f"style_tokens={n_style}"
        )

        # ── Generate: all conditions active from t=T ──────────────────────
        gen_result = gligen_pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, cartoon, duplicate",
            gligen_phrases=gligen_phrases,
            gligen_boxes=gligen_boxes,
            gligen_phrase_embeddings=phrase_embeds,
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=gligen_steps,
            height=512, width=512,
        )
        gen_img = gen_result.images[0]

        # ── Phase 2: Extract style tokens from shot 0 background ──────────
        if shot_id == 0 and has_style:
            all_bboxes = [e["box_xyxy"] for e in entities]
            bg_img     = extract_style_background(gen_img, all_bboxes)
            bg_path    = out_path / "shot_000_style_bg.png"
            bg_img.save(str(bg_path))
            with torch.no_grad():
                bg_clip      = entity_encoder.encode(bg_img).to(device)  # (1024,)
                style_tokens = style_encoder(bg_clip.unsqueeze(0)).squeeze(0)  # (K_g, 768)
            logger.info(
                f"  Style tokens extracted from shot 0 background "
                f"(K_g={style_tokens.shape[0]}, γ_style={style_weight})"
            )

        # ── GlobalStyle pixel blend (ablation only) ───────────────────────
        if use_style_blend and style_memory is not None:
            if style_memory.is_ready():
                gen_img = style_memory.apply(
                    gen_img, entity_bboxes=[e["box_xyxy"] for e in active]
                )
                logger.info(f"  Pixel-blend applied (ablation, shot {shot_id})")
            else:
                style_memory.initialize(gen_img)

        # ── Save ──────────────────────────────────────────────────────────
        from src.generation.pema_pipeline import _crop_bbox
        img_bbox  = draw_layout_on_image(gen_img.copy(), layout)
        shot_path = out_path / f"shot_{shot_id:03d}.png"
        bbox_path = out_path / f"shot_{shot_id:03d}_bbox.png"
        lo_path   = out_path / f"shot_{shot_id:03d}_layout_only.png"
        gen_img.save(str(shot_path))
        img_bbox.save(str(bbox_path))
        gen_img.save(str(lo_path))

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
            "shot_id":         shot_id,
            "path":            str(shot_path),
            "bbox_path":       str(bbox_path),
            "layout_only_path":str(lo_path),
            "active_entities": [e["name"] for e in active],
            "memory_updates":  update_log,
            "mode":            mode_full,
            "n_style_tokens":  n_style,
        })
        logger.info(
            f"  shot {shot_id} done | mode={mode_full} | memory: {update_log}"
        )

    return results
