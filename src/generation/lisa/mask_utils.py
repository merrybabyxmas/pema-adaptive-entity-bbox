"""
Gaussian soft mask generation for LISA regional conditioning.

Input: bounding boxes [(x1,y1,x2,y2)] on latent grid + sigma
Output: soft masks [N, H, W] with Gaussian falloff, summing to ~1.0
"""

import torch
import numpy as np
from PIL import Image


def create_gaussian_mask(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    sigma: float = 20.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a single Gaussian soft mask centered on bbox region.

    Args:
        bbox: (x1, y1, x2, y2) in latent coordinates
        height: latent height
        width: latent width
        sigma: Gaussian spread (larger = softer edges)
        device: torch device

    Returns:
        mask: (H, W) tensor with values in [0, 1]
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    mask = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    # Boost values inside bbox to ensure strong signal in the target region
    inside = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    mask = torch.where(inside, torch.clamp(mask + 0.5, max=1.0), mask)

    return mask


def create_entity_masks(
    bboxes: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    sigma: float = 20.0,
    device: str = "cpu",
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Create soft masks for multiple entities + background mask.

    Args:
        bboxes: list of (x1, y1, x2, y2) per entity
        height, width: latent dimensions
        sigma: Gaussian spread
        device: torch device

    Returns:
        entity_masks: list of (H, W) tensors, one per entity
        bg_mask: (H, W) background mask (complement of all entity masks)
    """
    entity_masks = []
    for bbox in bboxes:
        mask = create_gaussian_mask(bbox, height, width, sigma, device)
        entity_masks.append(mask)

    # Background mask: areas not covered by any entity
    if entity_masks:
        combined = torch.stack(entity_masks, dim=0).max(dim=0).values
        bg_mask = 1.0 - combined
        bg_mask = torch.clamp(bg_mask, min=0.0)
    else:
        bg_mask = torch.ones(height, width, device=device)

    return entity_masks, bg_mask


def normalize_masks(
    entity_masks: list[torch.Tensor],
    bg_mask: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Normalize all masks so they sum to 1.0 at every pixel.

    This ensures no signal amplification or loss during regional conditioning.
    """
    all_masks = torch.stack(entity_masks + [bg_mask], dim=0)
    total = all_masks.sum(dim=0, keepdim=True).clamp(min=1e-6)
    all_masks = all_masks / total

    normalized_entity = [all_masks[i] for i in range(len(entity_masks))]
    normalized_bg = all_masks[-1]
    return normalized_entity, normalized_bg


def masks_to_pil(
    entity_masks: list[torch.Tensor],
    bg_mask: torch.Tensor,
) -> list[Image.Image]:
    """Convert masks to PIL images for visualization."""
    images = []
    for mask in entity_masks + [bg_mask]:
        arr = (mask.cpu().numpy() * 255).astype(np.uint8)
        images.append(Image.fromarray(arr, mode="L"))
    return images


def bbox_from_layout(
    position: str,
    latent_h: int,
    latent_w: int,
    margin: float = 0.05,
) -> tuple[int, int, int, int]:
    """Generate bbox from a named position on the latent grid.

    Args:
        position: "left", "right", "center", "top-left", "top-right",
                  "bottom-left", "bottom-right"
        latent_h, latent_w: latent grid dimensions
        margin: fraction of margin from edges

    Returns:
        (x1, y1, x2, y2) in latent coordinates
    """
    mx = int(latent_w * margin)
    my = int(latent_h * margin)
    mid_x = latent_w // 2
    mid_y = latent_h // 2
    entity_w = int(latent_w * 0.4)
    entity_h = int(latent_h * 0.7)

    # Compact bboxes for 3-entity layouts — "wide shot" feel with clear separation
    third_w = int(latent_w * 0.22)
    third_h = int(latent_h * 0.55)
    third_x0 = mx + int(latent_w * 0.02)
    third_x1 = mid_x - third_w // 2
    third_x2 = latent_w - mx - third_w - int(latent_w * 0.02)

    positions = {
        "left": (mx, mid_y - entity_h // 2, mx + entity_w, mid_y + entity_h // 2),
        "right": (latent_w - mx - entity_w, mid_y - entity_h // 2, latent_w - mx, mid_y + entity_h // 2),
        "center": (mid_x - entity_w // 2, mid_y - entity_h // 2, mid_x + entity_w // 2, mid_y + entity_h // 2),
        "top-left": (mx, my, mx + entity_w, my + entity_h),
        "top-right": (latent_w - mx - entity_w, my, latent_w - mx, my + entity_h),
        "bottom-left": (mx, latent_h - my - entity_h, mx + entity_w, latent_h - my),
        "bottom-right": (latent_w - mx - entity_w, latent_h - my - entity_h, latent_w - mx, latent_h - my),
        # Compact 3-column positions with wide gaps
        "left-third": (third_x0, mid_y - third_h // 2, third_x0 + third_w, mid_y + third_h // 2),
        "center-third": (third_x1, mid_y - third_h // 2, third_x1 + third_w, mid_y + third_h // 2),
        "right-third": (third_x2, mid_y - third_h // 2, third_x2 + third_w, mid_y + third_h // 2),
    }

    if position not in positions:
        raise ValueError(f"Unknown position '{position}'. Choose from: {list(positions.keys())}")

    return positions[position]
