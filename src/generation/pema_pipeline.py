"""
PEMA: Presence-Aware Entity Memory Adapter generation pipeline.

Phase 1 — 2-pass + Global Style Memory:
  Pass 1:  GLIGEN text-box → base scene (correct bbox layout)
  Pass 2:  For each active entity: crop bbox → img2img with IP-Adapter (identity)
           → paste refined crop back (bbox-localized identity conditioning)
  Style:   GlobalStyleMemory initialized from shot 0 → background blend for shots 1+
  Memory:  CLIP EMA update of recent tokens (quality-gated)
"""
from __future__ import annotations
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

from src.memory.memory_bank import EntityMemoryBank
from src.generation.layout_adapter import plan_to_layout, draw_layout_on_image
from src.utils.logging import get_logger

logger = get_logger(__name__)

GLIGEN_MODEL      = "masterful/gligen-1-4-generation-text-box"
SD15_MODEL        = "runwayml/stable-diffusion-v1-5"
IP_ADAPTER_REPO   = "h94/IP-Adapter"
IP_ADAPTER_WEIGHT = "models/ip-adapter_sd15.bin"


# ─── Global Style Memory ──────────────────────────────────────────────────────

class GlobalStyleMemory:
    """
    Stores the first generated shot as a style anchor.
    Applied to subsequent shots via background blending:
      - entity bbox regions: kept from PEMA output (identity-refined)
      - non-entity regions:  blended with style anchor (scene/color consistency)
    """
    def __init__(self, blend_alpha: float = 0.45):
        self.anchor: Image.Image | None = None
        self.blend_alpha = blend_alpha  # weight of style anchor in background

    def initialize(self, image: Image.Image) -> None:
        self.anchor = image.copy()
        logger.info("GlobalStyleMemory initialized from shot 0")

    def is_ready(self) -> bool:
        return self.anchor is not None

    def apply(self, new_img: Image.Image, entity_bboxes: list[list[float]],
              feather: int = 24) -> Image.Image:
        """
        Blend style anchor background into new_img.
        Entity bbox regions are preserved from new_img.
        Background is alpha-blended with style anchor.
        """
        W, H = new_img.size
        anchor = self.anchor.resize((W, H), Image.LANCZOS)

        # Build entity mask: white = entity bbox (keep new_img), black = background (blend)
        entity_mask = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(entity_mask)
        for box in entity_bboxes:
            x1 = max(0, int(box[0] * W) - feather // 2)
            y1 = max(0, int(box[1] * H) - feather // 2)
            x2 = min(W, int(box[2] * W) + feather // 2)
            y2 = min(H, int(box[3] * H) + feather // 2)
            draw.ellipse([x1, y1, x2, y2], fill=255)
        entity_mask = entity_mask.filter(ImageFilter.GaussianBlur(radius=feather))

        # Background: blend anchor (for style) with new_img (for layout)
        bg_blend = Image.blend(anchor, new_img, 1.0 - self.blend_alpha)

        # Composite: entity regions from new_img, background from blended anchor
        result = Image.composite(new_img, bg_blend, entity_mask)
        return result


# ─── pipeline loaders ─────────────────────────────────────────────────────────

def _load_gligen(model_id: str, device: str):
    from diffusers import StableDiffusionGLIGENPipeline
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def _load_ip_adapter_img2img(model_id: str, device: str,
                              ip_repo: str = IP_ADAPTER_REPO,
                              ip_weight: str = IP_ADAPTER_WEIGHT,
                              ip_scale: float = 0.85):
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    pipe.load_ip_adapter(ip_repo, subfolder="models",
                         weight_name=ip_weight.split("/")[-1])
    pipe.set_ip_adapter_scale(ip_scale)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def _enhance_clip_for_ip_adapter(
    clip_emb: torch.Tensor,
    projector=None,
) -> torch.Tensor:
    """
    clip_emb: (1024,) float32 from CLIP ViT-H/14.
    Returns: (2, 1024) float16 for ip_adapter_image_embeds (doubled for CFG).

    ip_adapter_image_embeds bypasses CLIP encoding but still feeds through
    the UNet's encoder_hid_proj (ImageProjection: 1024→4×768), so we pass
    raw 1024d embeddings — optionally enhanced by the EntityProjector.
    """
    emb = clip_emb.unsqueeze(0)  # (1, 1024)
    if projector is not None:
        projector.eval()
        with torch.no_grad():
            emb = projector(emb.to(next(projector.parameters()).device)).to(emb.device)

    # Duplicate for CFG: [uncond_emb, cond_emb] — pipeline splits on chunk(2) call.
    # Unsqueeze to 3D (2, 1, 1024): diffusers check_inputs requires ndim ≥ 3.
    # ImageProjection: Linear(1024→3072) broadcasts over leading dims, then
    # reshape(B, 4, -1) → (2, 4, 768) for cross-attention.
    emb_16 = emb.to(torch.float16)
    emb_doubled = torch.cat([emb_16, emb_16], dim=0)  # (2, 1024)
    return emb_doubled.unsqueeze(1)  # (2, 1, 1024)


# ─── bbox deoverlap ───────────────────────────────────────────────────────────

def deoverlap_boxes(boxes_dict: dict[str, list[float]],
                    max_iter: int = 30) -> dict[str, list[float]]:
    """
    Iteratively push overlapping bboxes apart until they no longer overlap.
    Moves boxes along the axis of smallest penetration depth.
    boxes_dict: {entity_name: [x1,y1,x2,y2]} normalized [0,1]
    """
    names = list(boxes_dict.keys())
    if len(names) < 2:
        return boxes_dict

    boxes = [list(boxes_dict[n]) for n in names]

    for _ in range(max_iter):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                b1, b2 = boxes[i], boxes[j]
                ox1 = max(b1[0], b2[0])
                oy1 = max(b1[1], b2[1])
                ox2 = min(b1[2], b2[2])
                oy2 = min(b1[3], b2[3])
                if ox2 <= ox1 or oy2 <= oy1:
                    continue  # no overlap

                ov_w = ox2 - ox1
                ov_h = oy2 - oy1
                push = 0.02  # extra gap after separation

                if ov_w <= ov_h:
                    # push horizontally (smaller penetration axis)
                    c1x = (b1[0] + b1[2]) / 2
                    c2x = (b2[0] + b2[2]) / 2
                    half = ov_w / 2 + push
                    if c1x <= c2x:
                        boxes[i][0] -= half; boxes[i][2] -= half
                        boxes[j][0] += half; boxes[j][2] += half
                    else:
                        boxes[i][0] += half; boxes[i][2] += half
                        boxes[j][0] -= half; boxes[j][2] -= half
                else:
                    # push vertically
                    c1y = (b1[1] + b1[3]) / 2
                    c2y = (b2[1] + b2[3]) / 2
                    half = ov_h / 2 + push
                    if c1y <= c2y:
                        boxes[i][1] -= half; boxes[i][3] -= half
                        boxes[j][1] += half; boxes[j][3] += half
                    else:
                        boxes[i][1] += half; boxes[i][3] += half
                        boxes[j][1] -= half; boxes[j][3] -= half
                moved = True

        if not moved:
            break

    # clamp to valid image range
    for b in boxes:
        w = b[2] - b[0]
        h = b[3] - b[1]
        b[0] = max(0.02, min(0.98 - w, b[0]))
        b[2] = b[0] + w
        b[1] = max(0.02, min(0.98 - h, b[1]))
        b[3] = b[1] + h

    return {names[i]: boxes[i] for i in range(len(names))}


# ─── image helpers ────────────────────────────────────────────────────────────

def _crop_bbox(image: Image.Image, box_xyxy: list[float]) -> tuple[Image.Image, tuple]:
    W, H = image.size
    x1 = max(0, int(box_xyxy[0] * W))
    y1 = max(0, int(box_xyxy[1] * H))
    x2 = min(W, int(box_xyxy[2] * W))
    y2 = min(H, int(box_xyxy[3] * H))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return image, (0, 0, W, H)
    return image.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


def _paste_with_feather(base: Image.Image, crop: Image.Image,
                        coords: tuple, feather: int = 10) -> Image.Image:
    x1, y1, x2, y2 = coords
    crop_resized = crop.resize((x2 - x1, y2 - y1), Image.LANCZOS)

    mask = Image.new("L", crop_resized.size, 255)
    for i in range(feather):
        alpha = int(255 * i / feather)
        for x in range(crop_resized.width):
            mask.putpixel((x, i), min(mask.getpixel((x, i)), alpha))
            row = crop_resized.height - 1 - i
            mask.putpixel((x, row), min(mask.getpixel((x, row)), alpha))
        for y in range(crop_resized.height):
            mask.putpixel((i, y), min(mask.getpixel((i, y)), alpha))
            col = crop_resized.width - 1 - i
            mask.putpixel((col, y), min(mask.getpixel((col, y)), alpha))

    result = base.copy()
    result.paste(crop_resized, (x1, y1), mask)
    return result


# ─── bootstrap ────────────────────────────────────────────────────────────────

_BOOTSTRAP_NEG = (
    "watermark, text, logo, signature, blurry, low quality, "
    "cartoon, illustration, painting, multiple animals"
)

def bootstrap_entity_image(entity_name: str, gligen_pipe,
                            bootstrap_dir: Path, steps: int = 40) -> Image.Image:
    cache = bootstrap_dir / f"{entity_name}_bootstrap.png"
    if cache.exists():
        logger.info(f"  bootstrap cache hit: {entity_name}")
        return Image.open(str(cache)).convert("RGB")

    prompt = (
        f"A single {entity_name}, full body visible, "
        f"plain white background, high quality photograph, sharp focus"
    )
    result = gligen_pipe(
        prompt=prompt,
        negative_prompt=_BOOTSTRAP_NEG,
        gligen_phrases=[entity_name],
        gligen_boxes=[[0.15, 0.15, 0.85, 0.85]],
        gligen_scheduled_sampling_beta=1.0,
        num_inference_steps=steps,
        height=512, width=512,
    )
    img = result.images[0]
    img.save(str(cache))
    logger.info(f"  bootstrap generated: {entity_name} → {cache.name}")
    return img


# ─── main generation ──────────────────────────────────────────────────────────

def generate_with_pema(
    plan_output: dict,
    memory_bank: EntityMemoryBank,
    output_dir: str,
    gligen_pipe=None,
    ip_pipe=None,
    style_memory: GlobalStyleMemory | None = None,
    device: str = "cuda",
    update_memory: bool = True,
    gligen_steps: int = 30,
    refine_steps: int = 25,
    refine_strength: float = 0.75,
    ip_scale: float = 0.85,
    projector=None,
) -> list[dict]:
    """
    PEMA Phase 1+2 generation.

    Pass 1:  GLIGEN text-box → base scene
    Pass 2:  entity bbox crop → IP-Adapter img2img (identity) → paste back
             If projector provided: CLIP embedding enhanced via EntityProjector
             and injected as ip_adapter_image_embeds (Phase 2 mode).
             Otherwise: standard ip_adapter_image path (Phase 1 mode).
    Style:   GlobalStyleMemory background blend for shots 1+
    Memory:  CLIP EMA update from crop (quality-gated)
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layouts = plan_to_layout(plan_output)

    if gligen_pipe is None:
        gligen_pipe = _load_gligen(GLIGEN_MODEL, device)
        logger.info(f"Loaded GLIGEN: {GLIGEN_MODEL}")

    if ip_pipe is None:
        ip_pipe = _load_ip_adapter_img2img(SD15_MODEL, device, ip_scale=ip_scale)
        logger.info("Loaded IP-Adapter img2img")
    else:
        ip_pipe.set_ip_adapter_scale(ip_scale)

    if style_memory is None:
        style_memory = GlobalStyleMemory()

    results = []
    for layout in layouts:
        shot_id = layout["shot_id"]
        prompt = layout["prompt"]
        entities = layout["entities"]
        active = [e for e in entities if memory_bank.has(e["name"])]

        # ── Deoverlap bboxes before GLIGEN ────────────────────────────────
        raw_boxes = {e["name"]: e["box_xyxy"] for e in entities}
        deov_boxes = deoverlap_boxes(raw_boxes)
        for e in entities:
            if e["name"] in deov_boxes:
                orig = e["box_xyxy"]
                fixed = deov_boxes[e["name"]]
                if fixed != orig:
                    logger.info(
                        f"  deoverlap {e['name']}: "
                        f"[{orig[0]:.2f},{orig[1]:.2f},{orig[2]:.2f},{orig[3]:.2f}] → "
                        f"[{fixed[0]:.2f},{fixed[1]:.2f},{fixed[2]:.2f},{fixed[3]:.2f}]"
                    )
                e["box_xyxy"] = fixed

        # ── Pass 1: GLIGEN layout ──────────────────────────────────────────
        all_phrases = [e["name"] for e in entities]
        all_boxes   = [e["box_xyxy"] for e in entities]

        logger.info(f"Shot {shot_id} | Pass1 GLIGEN: {all_phrases}")
        p1 = gligen_pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, cartoon, duplicate entities",
            gligen_phrases=all_phrases,
            gligen_boxes=all_boxes,
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=gligen_steps,
            height=512, width=512,
        )
        base_img = p1.images[0]

        # ── Pass 2: bbox-localized IP-Adapter identity refinement ──────────
        refined_img = base_img.copy()
        for e in active:
            crop, coords = _crop_bbox(base_img, e["box_xyxy"])
            if crop.size[0] < 16 or crop.size[1] < 16:
                continue

            crop_512 = crop.resize((512, 512), Image.LANCZOS)

            if projector is not None:
                # Phase 2: EntityProjector enhances CLIP embedding → ip_adapter_image_embeds
                # ip_adapter_image_embeds bypasses CLIP encoding but still goes through
                # UNet's ImageProjection (1024d → 4×768 tokens), so we pass 1024d embeds.
                clip_emb = memory_bank.retrieve_tokens(e["name"])  # (1024,) float32
                enhanced = _enhance_clip_for_ip_adapter(
                    clip_emb.to(device), projector
                )  # (2, 1024) float16 — doubled for CFG
                p2 = ip_pipe(
                    prompt=f"a {e['name']}, photorealistic, high quality",
                    negative_prompt="blurry, low quality, cartoon, watermark",
                    image=crop_512,
                    ip_adapter_image_embeds=[enhanced],
                    strength=refine_strength,
                    num_inference_steps=refine_steps,
                )
                mode_tag = "phase2"
            else:
                # Phase 1: standard image-based IP-Adapter
                ref_img = memory_bank.retrieve_image(e["name"])
                p2 = ip_pipe(
                    prompt=f"a {e['name']}, photorealistic, high quality",
                    negative_prompt="blurry, low quality, cartoon, watermark",
                    image=crop_512,
                    ip_adapter_image=ref_img,
                    strength=refine_strength,
                    num_inference_steps=refine_steps,
                )
                mode_tag = "phase1"

            refined_crop = p2.images[0]
            refined_img = _paste_with_feather(refined_img, refined_crop, coords)
            logger.info(f"  Pass2 [{mode_tag}] identity: {e['name']} at {coords}")

        # ── Global Style Memory: background consistency ────────────────────
        if style_memory.is_ready():
            styled_img = style_memory.apply(
                refined_img,
                entity_bboxes=[e["box_xyxy"] for e in active],
            )
            logger.info(f"  GlobalStyle applied (shot {shot_id})")
        else:
            styled_img = refined_img
            style_memory.initialize(refined_img)  # shot 0 → anchor

        # ── Save outputs ──────────────────────────────────────────────────
        img_bbox = draw_layout_on_image(styled_img.copy(), layout)
        shot_path  = out_path / f"shot_{shot_id:03d}.png"
        bbox_path  = out_path / f"shot_{shot_id:03d}_bbox.png"
        base_path  = out_path / f"shot_{shot_id:03d}_base.png"
        styled_img.save(str(shot_path))
        img_bbox.save(str(bbox_path))
        base_img.save(str(base_path))

        # ── Memory update ─────────────────────────────────────────────────
        update_log = {}
        if update_memory:
            for e in active:
                crop, _ = _crop_bbox(styled_img, e["box_xyxy"])
                crop_path = out_path / f"shot_{shot_id:03d}_crop_{e['name']}.png"
                crop.save(str(crop_path))
                accepted = memory_bank.update(e["name"], crop)
                update_log[e["name"]] = "updated" if accepted else "rejected"

        results.append({
            "shot_id": shot_id,
            "path": str(shot_path),
            "bbox_path": str(bbox_path),
            "base_path": str(base_path),
            "active_entities": [e["name"] for e in active],
            "memory_updates": update_log,
        })
        logger.info(f"  shot {shot_id} done | memory: {update_log}")

    return results
