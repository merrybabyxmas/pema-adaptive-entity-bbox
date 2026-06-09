"""Layout-guided generation pipeline — supports plain SD, GLIGEN, and GLIGEN+ref images."""
import torch
from pathlib import Path
from PIL import Image

from src.generation.layout_adapter import plan_to_layout, draw_layout_on_image
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _load_sd_pipe(model_id, device):
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    )
    return pipe.to(device)


def _load_gligen_pipe(model_id, device):
    from diffusers import StableDiffusionGLIGENPipeline
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    return pipe.to(device)


def generate_entity_references(entities: list[str], pipe, ref_dir: Path,
                                steps: int = 30) -> dict[str, Image.Image]:
    """
    Generate one reference image per entity if not already cached.
    Returns {entity_name: PIL.Image}.
    """
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    for entity in entities:
        cache_path = ref_dir / f"{entity}.png"
        if cache_path.exists():
            refs[entity] = Image.open(cache_path).convert("RGB")
            logger.info(f"  ref loaded from cache: {entity}")
            continue
        # Generate single-entity image with GLIGEN (centered bbox)
        result = pipe(
            prompt=f"a photo of a {entity}, natural background, high quality",
            gligen_phrases=[entity],
            gligen_boxes=[[0.1, 0.1, 0.9, 0.9]],
            gligen_scheduled_sampling_beta=1.0,
            num_inference_steps=steps,
            height=512, width=512,
        )
        img = result.images[0]
        img.save(str(cache_path))
        refs[entity] = img
        logger.info(f"  ref generated & cached: {entity} -> {cache_path.name}")
    return refs


def generate_with_layout(plan_output: dict, output_dir: str,
                         use_sd: bool = True,
                         sd_model_id: str = "runwayml/stable-diffusion-v1-5",
                         use_gligen: bool = False,
                         gligen_model_id: str = "masterful/gligen-1-4-generation-text-box",
                         use_ref: bool = False,
                         ref_dir: str = None,
                         device: str = "cuda"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    layouts = plan_to_layout(plan_output)
    all_entities = list({e["name"] for l in layouts for e in l["entities"]})

    pipe = None
    mode = "blank"
    entity_refs = {}

    if use_gligen or use_ref:
        try:
            pipe = _load_gligen_pipe(gligen_model_id, device)
            pipe.set_progress_bar_config(disable=True)
            mode = "gligen"
            logger.info(f"GLIGEN pipeline loaded: {gligen_model_id}")

            if use_ref:
                _ref_dir = Path(ref_dir) if ref_dir else out_path.parent / "entity_refs"
                entity_refs = generate_entity_references(all_entities, pipe, _ref_dir)
        except Exception as e:
            logger.warning(f"GLIGEN load failed: {e}, falling back to SD")
            use_gligen = False
            use_ref = False
            use_sd = True

    if mode == "blank" and use_sd:
        try:
            pipe = _load_sd_pipe(sd_model_id, device)
            pipe.set_progress_bar_config(disable=True)
            mode = "sd"
            logger.info(f"SD pipeline loaded: {sd_model_id}")
        except Exception as e:
            logger.warning(f"SD load failed: {e}, falling back to blank canvas")

    images = []
    for layout in layouts:
        prompt = layout["prompt"]
        logger.info(f"Generating shot {layout['shot_id']}: {prompt}")

        if mode == "gligen":
            phrases = [e["name"] for e in layout["entities"]]
            boxes = [e["box_xyxy"] for e in layout["entities"]]
            kwargs = dict(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_boxes=boxes,
                gligen_scheduled_sampling_beta=1.0,
                num_inference_steps=30,
                height=512, width=512,
            )
            # attach reference images if available
            if use_ref and entity_refs:
                ref_imgs = [entity_refs.get(p) for p in phrases]
                if any(r is not None for r in ref_imgs):
                    # replace missing refs with a blank white image
                    blank = Image.new("RGB", (512, 512), (255, 255, 255))
                    kwargs["gligen_images"] = [r if r is not None else blank for r in ref_imgs]

            result = pipe(**kwargs)
            img = result.images[0]

        elif mode == "sd":
            result = pipe(prompt, num_inference_steps=30, height=512, width=512)
            img = result.images[0]

        else:
            img = Image.new("RGB", (512, 512), color=(200, 200, 200))

        img_with_boxes = draw_layout_on_image(img.copy(), layout)
        shot_path = out_path / f"shot_{layout['shot_id']:03d}.png"
        shot_bbox_path = out_path / f"shot_{layout['shot_id']:03d}_bbox.png"
        img.save(shot_path)
        img_with_boxes.save(shot_bbox_path)
        images.append({"shot_id": layout["shot_id"], "path": str(shot_path),
                       "bbox_path": str(shot_bbox_path)})
        logger.info(f"  saved: {shot_path}")

    return images
