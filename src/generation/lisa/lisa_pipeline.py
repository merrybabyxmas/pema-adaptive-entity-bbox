"""
LISA Pipeline: Layout-Integrated Spatial Anchoring.

Phase 3+4 combined: Region-Guided Decoupled Conditioning with Global Self-Attention.

Strategy:
- Use diffusers native IP-Adapter with multiple images + ip_adapter_masks
- Each entity gets its own IP-Adapter image with a spatial Gaussian soft mask
- Self-Attention remains global (natural harmonization)
- Cross-Attention is regionally weighted via masks

Usage:
    pipeline = LISAPipeline(config)
    pipeline.load_models()
    results = pipeline.generate_shot(anchor_bank, layout_plan, shot_index=0)
"""

import os
import sys
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation.lisa.mask_utils import create_entity_masks, normalize_masks


class LISAPipeline:
    """Main LISA generation pipeline using native diffusers IP-Adapter mask support."""

    def __init__(self, config: dict):
        self.config = config
        self.gen_cfg = config["generation"]
        self.cond_cfg = config["conditioning"]
        self.layout_cfg = config["layout"]
        self.device = torch.device(self.gen_cfg["device"])
        self.dtype = torch.float16 if self.gen_cfg["dtype"] == "float16" else torch.float32

        self.pipe = None

    def load_models(self):
        """Load SDXL + IP-Adapter pipeline."""
        from diffusers import StableDiffusionXLPipeline

        print("[LISA] Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.config["models"]["sdxl"],
            torch_dtype=self.dtype,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.to(self.device)
        self.pipe.vae.enable_slicing()

        # Load IP-Adapter
        ip_cfg = self.config["models"]["ip_adapter"]
        print("[LISA] Loading IP-Adapter...")
        self.pipe.load_ip_adapter(
            ip_cfg["repo"],
            subfolder=ip_cfg["subfolder"],
            weight_name=ip_cfg["weight_name"],
            image_encoder_folder=ip_cfg["image_encoder"],
        )

        print("[LISA] Models loaded successfully.")

    def _prepare_ip_masks(
        self,
        entity_masks: list[torch.Tensor],
        output_h: int,
        output_w: int,
    ) -> list[torch.Tensor]:
        """Prepare IP-Adapter masks from latent-space entity masks.

        Resizes Gaussian soft masks to output resolution and formats
        them for diffusers' ip_adapter_masks parameter.

        Returns:
            list with one tensor of shape (1, num_entities, H, W)
        """
        ip_masks = []
        for m in entity_masks:
            resized = F.interpolate(
                m.unsqueeze(0).unsqueeze(0).float(),
                size=(output_h, output_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # (1, H, W)
            ip_masks.append(resized)

        # Shape: [1, num_entities, H, W] — one mask per IP image
        stacked = torch.cat(ip_masks, dim=0).unsqueeze(0)
        return [stacked]

    def generate_shot(
        self,
        anchor_bank: dict,
        layout_plan: dict,
        shot_index: int = 0,
        seed: int | None = None,
        sigma_override: float | None = None,
        ip_scale_override: float | None = None,
    ) -> dict:
        """Generate a single shot with regional entity conditioning.

        Uses diffusers native IP-Adapter masks for spatial control:
        - Each entity anchor image gets its own Gaussian soft mask
        - IP-Adapter contributions are spatially weighted per-entity
        - Self-Attention remains global for natural harmonization
        """
        if self.pipe is None:
            raise RuntimeError("Call load_models() first")

        shot = layout_plan["shots"][shot_index]
        latent_h = self.layout_cfg["latent_size"]
        latent_w = self.layout_cfg["latent_size"]
        sigma = sigma_override if sigma_override is not None else self.layout_cfg["mask_sigma"]

        # --- Build entity masks ---
        bboxes = [tuple(ent["bbox"]) for ent in shot["entities"]]
        entity_masks, bg_mask = create_entity_masks(bboxes, latent_h, latent_w, sigma, self.device)
        entity_masks, bg_mask = normalize_masks(entity_masks, bg_mask)

        # --- Load entity anchor images ---
        entity_images = []
        entity_name_to_anchor = {a["entity_name"]: a for a in anchor_bank["entities"]}

        for ent_info in shot["entities"]:
            anchor = entity_name_to_anchor[ent_info["name"]]
            img = Image.open(anchor["image_path"]).convert("RGB")
            entity_images.append(img)

        # --- IP-Adapter config ---
        ip_adapter_scale = ip_scale_override if ip_scale_override is not None else self.cond_cfg["ip_adapter_scale"]
        self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        # --- Prepare spatial masks for IP-Adapter ---
        ip_adapter_masks = self._prepare_ip_masks(
            entity_masks,
            self.gen_cfg["height"],
            self.gen_cfg["width"],
        )

        # --- Build combined prompt ---
        entity_descs = []
        for ent_info in shot["entities"]:
            ent_def = next(
                e for e in layout_plan["entity_definitions"]
                if e["name"] == ent_info["name"]
            )
            pos = ent_info["position"]
            entity_descs.append(f"{ent_def['prompt']} on the {pos}")

        bg_def = layout_plan["background"]
        full_prompt = (
            f"{', '.join(entity_descs)}, "
            f"{bg_def['prompt']}, "
            f"{shot['description']}, "
            f"high quality, detailed, 4k photograph"
        )
        negative_prompt = (
            "blurry, low quality, deformed, ugly, duplicate, "
            "merged figures, identity mixing, chimera, "
            "extra limbs, disfigured"
        )

        # --- Generate ---
        gen_seed = seed if seed is not None else self.gen_cfg["seed"]
        generator = torch.Generator(device=self.device).manual_seed(gen_seed)

        print(f"[LISA] Generating shot {shot_index}: {shot['description'][:60]}...")
        print(f"  Entities: {[e['name'] for e in shot['entities']]}")
        print(f"  IP-Adapter masks: {ip_adapter_masks[0].shape}")

        result = self.pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=[entity_images],
            cross_attention_kwargs={"ip_adapter_masks": ip_adapter_masks},
            height=self.gen_cfg["height"],
            width=self.gen_cfg["width"],
            num_inference_steps=self.gen_cfg["num_inference_steps"],
            guidance_scale=self.gen_cfg["guidance_scale"],
            generator=generator,
        )

        image = result.images[0]

        # Save output
        output_dir = self.config["evaluation"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"shot_{shot_index:03d}.png")
        image.save(save_path)
        print(f"  Saved: {save_path}")

        return {
            "image": image,
            "shot_index": shot_index,
            "save_path": save_path,
            "prompt": full_prompt,
        }

    def generate_all_shots(
        self,
        anchor_bank: dict,
        layout_plan: dict,
        seeds: list[int] | None = None,
    ) -> list[dict]:
        """Generate all shots in a layout plan."""
        results = []
        for i in range(len(layout_plan["shots"])):
            seed = seeds[i] if seeds else self.gen_cfg["seed"] + i
            result = self.generate_shot(anchor_bank, layout_plan, shot_index=i, seed=seed)
            results.append(result)
        return results


def load_pipeline(config_path: str = None) -> LISAPipeline:
    """Convenience function to load LISA pipeline from config."""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "configs" / "default.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipeline = LISAPipeline(config)
    pipeline.load_models()
    return pipeline
