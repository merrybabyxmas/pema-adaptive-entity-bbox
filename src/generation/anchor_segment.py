"""
Segment the entity out of an anchor crop (SAM, box prompt) and composite it onto
a neutral background, so the IP-Adapter image tokens encode OBJECT IDENTITY only
— not the anchor's specific background, which otherwise leaks into the generated
scene and looks unnatural.
"""
from __future__ import annotations
import numpy as np
import torch
from PIL import Image


class AnchorSegmenter:
    def __init__(self, device="cuda", model_id="facebook/sam-vit-base"):
        from transformers import SamModel, SamProcessor
        self.device = device
        self.model = SamModel.from_pretrained(model_id).to(device).eval()
        self.processor = SamProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def mask(self, img: Image.Image, inset: float = 0.08):
        """Return binary object mask (H,W) float32 via box-prompted SAM."""
        img = img.convert("RGB"); W, H = img.size
        box = [[[W*inset, H*inset, W*(1-inset), H*(1-inset)]]]
        inp = self.processor(img, input_boxes=box, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0][0]
        iou = out.iou_scores.cpu()[0, 0]
        m = masks[int(iou.argmax())].numpy().astype("float32")
        return m if m.mean() >= 0.03 else None

    @torch.no_grad()
    def segment(self, img: Image.Image, bg: int = 128, inset: float = 0.08) -> Image.Image:
        """Box-prompted SAM (the anchor crop is object-centric) → object on a
        flat neutral-gray background. Returns RGB PIL same size as input."""
        img = img.convert("RGB")
        W, H = img.size
        box = [[[W*inset, H*inset, W*(1-inset), H*(1-inset)]]]
        inp = self.processor(img, input_boxes=box, return_tensors="pt").to(self.device)
        out = self.model(**inp)
        masks = self.processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inp["original_sizes"].cpu(),
            inp["reshaped_input_sizes"].cpu())[0][0]
        iou = out.iou_scores.cpu()[0, 0]
        m = masks[int(iou.argmax())].numpy().astype(np.float32)   # (H,W)
        if m.mean() < 0.03:        # segmentation failed → return original
            return img
        arr = np.asarray(img).astype(np.float32)
        comp = arr * m[..., None] + bg * (1 - m[..., None])
        return Image.fromarray(comp.astype("uint8"))
