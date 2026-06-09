"""
Entity encoder using IP-Adapter's CLIP ViT-H/14 (projection_dim=1024).
Matches the embedding space expected by IP-Adapter's ImageProjection.
"""
from __future__ import annotations
import torch
from pathlib import Path
from PIL import Image


_IP_ADAPTER_CACHE = (
    Path.home() / ".cache/huggingface/hub"
    / "models--h94--IP-Adapter/snapshots"
)


def _find_image_encoder_path() -> str:
    snapshots = list(_IP_ADAPTER_CACHE.iterdir())
    if not snapshots:
        raise FileNotFoundError("IP-Adapter not cached. Run generate_pema.py first.")
    return str(snapshots[0] / "models/image_encoder")


class EntityEncoder:
    """
    Encodes entity images using CLIP ViT-H/14 (same encoder as IP-Adapter).
    Output: 1024d pooled projection vector — directly compatible with
            IP-Adapter's ImageProjection layer.
    """

    def __init__(self, device: str = "cuda"):
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        enc_path = _find_image_encoder_path()
        self.device = device
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            enc_path, torch_dtype=torch.float16
        ).to(device).eval()

        # IP-Adapter's image_encoder directory lacks preprocessor_config.json.
        # Fall back to standard CLIP ViT-H/14 preprocessing parameters.
        proc_config = Path(enc_path) / "preprocessor_config.json"
        if proc_config.exists():
            self.processor = CLIPImageProcessor.from_pretrained(enc_path)
        else:
            self.processor = CLIPImageProcessor(
                size={"shortest_edge": 224},
                crop_size={"height": 224, "width": 224},
                do_resize=True,
                do_center_crop=True,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711],
                resample=3,
                do_convert_rgb=True,
            )

    @torch.no_grad()
    def encode(self, image: Image.Image) -> torch.Tensor:
        """Returns (1024,) float32 tensor."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        return out.image_embeds.squeeze(0).float()  # (1024,)

    @torch.no_grad()
    def encode_patches(self, image: Image.Image, grid: int = 4) -> torch.Tensor:
        """
        Richer entity tokens (IP-Adapter-Plus style): penultimate CLIP hidden
        states carry spatial identity detail that the single pooled embedding
        discards. Drop CLS, reshape the 16x16 patch grid, adaptive-avg-pool to
        (grid x grid), and return (grid*grid, 1280) region tokens.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        patches = out.hidden_states[-2][:, 1:, :]      # (1, 256, 1280) drop CLS
        B, N, D = patches.shape
        side = int(N ** 0.5)                           # 16
        grid_feat = patches.reshape(B, side, side, D).permute(0, 3, 1, 2)
        pooled = torch.nn.functional.adaptive_avg_pool2d(grid_feat, (grid, grid))
        pooled = pooled.flatten(2).transpose(1, 2)     # (1, grid*grid, 1280)
        return pooled.squeeze(0).float()               # (grid*grid, 1280)

    def similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        a = a / (a.norm() + 1e-8)
        b = b / (b.norm() + 1e-8)
        return (a * b).sum().item()
