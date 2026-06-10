"""
Phase 1: Anchor Bank Construction.

Builds entity anchors (IP-Adapter embeddings on white background)
and background anchors (clean scene images) for LISA pipeline.
"""

import os
import sys
import torch
import yaml
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_entity_anchor(
    pipe,
    entity_prompt: str,
    entity_name: str,
    save_dir: str,
    config: dict,
) -> dict:
    """Generate entity on white background and extract IP-Adapter embedding.

    Args:
        pipe: SDXL pipeline (without IP-Adapter for generation)
        entity_prompt: description of the entity (e.g. "a blonde woman in red dress")
        entity_name: identifier (e.g. "entity_A")
        save_dir: directory to save anchor files
        config: configuration dict

    Returns:
        dict with 'image_path' and 'entity_name'
    """
    gen_cfg = config["generation"]
    anchor_cfg = config["anchors"]

    full_prompt = f"a single {entity_prompt}, solo, alone, {anchor_cfg['entity_prompt_suffix']}"
    # steer SDXL away from contact-sheet / sticker-collage of many instances
    anchor_negative = (
        "multiple, many, group, collage, grid, contact sheet, duplicate, "
        "repeated, several, two, three, tiled, montage, blurry, low quality"
    )

    generator = torch.Generator(device=gen_cfg["device"]).manual_seed(gen_cfg["seed"])
    image = pipe(
        prompt=full_prompt,
        negative_prompt=anchor_negative,
        height=gen_cfg["height"],
        width=gen_cfg["width"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=gen_cfg["guidance_scale"],
        generator=generator,
    ).images[0]

    # Save anchor image
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{entity_name}.png")
    image.save(image_path)

    return {
        "entity_name": entity_name,
        "image_path": image_path,
    }


def build_background_anchor(
    pipe,
    bg_prompt: str,
    bg_name: str,
    save_dir: str,
    config: dict,
) -> dict:
    """Generate clean background image.

    Args:
        pipe: SDXL pipeline
        bg_prompt: description of background (e.g. "a serene park with trees, no people")
        bg_name: identifier (e.g. "bg_park")
        save_dir: directory to save
        config: configuration dict

    Returns:
        dict with 'image_path' and 'bg_name'
    """
    gen_cfg = config["generation"]

    full_prompt = f"{bg_prompt}, empty scene, no people, no characters, high quality photograph"

    generator = torch.Generator(device=gen_cfg["device"]).manual_seed(gen_cfg["seed"])
    image = pipe(
        prompt=full_prompt,
        height=gen_cfg["height"],
        width=gen_cfg["width"],
        num_inference_steps=gen_cfg["num_inference_steps"],
        guidance_scale=gen_cfg["guidance_scale"],
        generator=generator,
    ).images[0]

    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"{bg_name}.png")
    image.save(image_path)

    return {
        "bg_name": bg_name,
        "image_path": image_path,
    }


def build_all_anchors(
    pipe,
    entities: list[dict],
    background: dict,
    config: dict,
) -> dict:
    """Build complete anchor bank for a scenario.

    Args:
        pipe: SDXL pipeline
        entities: list of {"name": str, "prompt": str}
        background: {"name": str, "prompt": str}
        config: configuration dict

    Returns:
        anchor_bank: {
            "entities": [{"entity_name", "image_path"}, ...],
            "background": {"bg_name", "image_path"},
        }
    """
    save_dir = config["anchors"]["save_dir"]

    entity_anchors = []
    for ent in entities:
        anchor = build_entity_anchor(
            pipe=pipe,
            entity_prompt=ent["prompt"],
            entity_name=ent["name"],
            save_dir=save_dir,
            config=config,
        )
        entity_anchors.append(anchor)
        print(f"  Built entity anchor: {ent['name']} -> {anchor['image_path']}")

    bg_anchor = build_background_anchor(
        pipe=pipe,
        bg_prompt=background["prompt"],
        bg_name=background["name"],
        save_dir=save_dir,
        config=config,
    )
    print(f"  Built background anchor: {background['name']} -> {bg_anchor['image_path']}")

    return {
        "entities": entity_anchors,
        "background": bg_anchor,
    }
