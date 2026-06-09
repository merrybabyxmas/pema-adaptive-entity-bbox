"""
MemoryGLIGENPipeline: extends StableDiffusionGLIGENPipeline to accept
pre-computed or memory-enhanced grounding phrase embeddings.

Key change: adds `gligen_phrase_embeddings` parameter to __call__.
When provided, replaces text_encoder.pooler_output in the grounding step
(diffusers StableDiffusionGLIGENPipeline L187) with entity memory tokens.

This is the correct conditioning approach: entity memory enters the
generation at the very first denoising step via GLIGEN's gated
self-attention (bbox-localized), not as post-hoc crop refinement.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import StableDiffusionGLIGENPipeline


class MemoryGLIGENPipeline(StableDiffusionGLIGENPipeline):
    """
    GLIGEN pipeline with entity memory conditioning support.

    gligen_phrase_embeddings: Tensor of shape (n_entities, 768) containing
    memory-enhanced grounding embeddings. When provided, replaces the
    text_encoder pooler_output for grounding tokens.

    Detection: grounding tokenizer call uses padding=True (short seq),
    while main prompt call uses max_length=77. We detect by seq_len < max_len.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        gligen_scheduled_sampling_beta: float = 0.3,
        gligen_phrases: List[str] = None,
        gligen_boxes: List[List[float]] = None,
        gligen_inpaint_image=None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator=None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        # ── New: entity memory grounding embeddings ───────────────────────
        gligen_phrase_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        gligen_phrase_embeddings: (n_entities, 768) float tensor.
          Pre-computed grounding embeddings that replace the text encoder
          pooler_output for GLIGEN's bbox-localized gated self-attention.
          Shape must match len(gligen_phrases).

          Typically: text_enc_pooler + EntityMemoryConditioner(M_e)
        """
        if gligen_phrase_embeddings is None:
            # No memory injection — fall back to standard GLIGEN
            return super().__call__(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
                gligen_phrases=gligen_phrases, gligen_boxes=gligen_boxes,
                gligen_inpaint_image=gligen_inpaint_image,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta, generator=generator, latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type, return_dict=return_dict,
                callback=callback, callback_steps=callback_steps,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_skip=clip_skip,
            )

        # ── Patch text_encoder.forward to intercept grounding call ────────
        # The grounding call uses padding=True → short seq_len < 77.
        # The main prompt call uses max_length=77. We distinguish by this.
        max_len = self.tokenizer.model_max_length
        device = self._execution_device
        dtype = self.text_encoder.dtype

        original_forward = self.text_encoder.forward
        target_embeds = gligen_phrase_embeddings.to(device=device, dtype=dtype)

        def _patched_forward(input_ids=None, **kwargs):
            result = original_forward(input_ids=input_ids, **kwargs)
            # Grounding call: seq_len < max_len (phrases are short)
            if input_ids is not None and input_ids.shape[1] < max_len:
                result.pooler_output = target_embeds
            return result

        self.text_encoder.forward = _patched_forward
        try:
            out = super().__call__(
                prompt=prompt, height=height, width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta,
                gligen_phrases=gligen_phrases, gligen_boxes=gligen_boxes,
                gligen_inpaint_image=gligen_inpaint_image,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                eta=eta, generator=generator, latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type, return_dict=return_dict,
                callback=callback, callback_steps=callback_steps,
                cross_attention_kwargs=cross_attention_kwargs,
                clip_skip=clip_skip,
            )
        finally:
            self.text_encoder.forward = original_forward

        return out
