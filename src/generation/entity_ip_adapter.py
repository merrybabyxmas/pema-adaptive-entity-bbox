"""
v14: Presence-aware BBox-localized Entity Image Adapter.

Reuses the *trained* IP-Adapter (SD1.5) decoupled image cross-attention
(to_k_ip / to_v_ip per attn2 layer + the image projection) — NO training — but
routes it per entity:

  z = z_text + Σ_{e ∈ active}  M_{bbox_e} ⊙ scale · Attn(Q, K_e^img, V_e^img)

  K_e^img, V_e^img = to_k_ip/to_v_ip( image_proj( CLIP(ref_e) ) )

vs vanilla IP-Adapter (one global image cond), this gives:
  - presence-aware: only entities present in the shot contribute,
  - bbox-localized: each entity's image attention is masked to its own bbox,
  - multi-entity: no cross-entity / background leakage.

Usage: extract_ip_adapter() to pull trained weights from a load_ip_adapter'd
pipeline, then install_entity_ip() on the target UNet + drive via the controller.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def bbox_mask(bbox, H, W, device, pad=0.05):
    x1, y1, x2, y2 = bbox
    x1 = max(0., x1 - pad); y1 = max(0., y1 - pad)
    x2 = min(1., x2 + pad); y2 = min(1., y2 + pad)
    ys = (torch.arange(H, device=device).float() + 0.5) / H
    xs = (torch.arange(W, device=device).float() + 0.5) / W
    m = ((ys >= y1) & (ys < y2)).view(H, 1) & ((xs >= x1) & (xs < x2)).view(1, W)
    return m.view(H * W)


def bbox_mask_feathered(bbox, H, W, device, pad=0.05, feather=0.08):
    """Soft bbox mask in [0,1] with a smooth falloff at the edges — avoids the
    hard seam between adjacent entity regions that makes the background look
    stitched."""
    x1, y1, x2, y2 = bbox
    x1 = max(0., x1 - pad); y1 = max(0., y1 - pad)
    x2 = min(1., x2 + pad); y2 = min(1., y2 + pad)
    ys = (torch.arange(H, device=device).float() + 0.5) / H
    xs = (torch.arange(W, device=device).float() + 0.5) / W
    f = max(feather, 1e-4)
    def band(c, lo, hi):
        return (torch.clamp((c - lo) / f, 0, 1) * torch.clamp((hi - c) / f, 0, 1)).clamp(0, 1)
    mx = band(xs, x1, x2).view(1, W)
    my = band(ys, y1, y2).view(H, 1)
    return (mx * my).reshape(H * W)


class EntityIPController:
    """Per-shot state: active entities' projected image tokens + bboxes.

    t_apply_below: only inject IP when the current timestep < this (∈[0,1000]).
    The scene/background structure is laid by text+GLIGEN in the early (high-t)
    steps; injecting identity only in later (low-t) steps refines the object's
    appearance WITHOUT restructuring the background → coherent, seam-free scene.
    feather: soft bbox edges (also reduces the inter-region seam).
    """
    def __init__(self, scale=1.0, cfg=True, t_apply_below=1000.0, feather=0.08):
        self.scale = scale
        self.cfg = cfg
        self.t_apply_below = t_apply_below
        self.feather = feather
        self.cur_t = None
        self.active = []     # list of (entity_name, tokens[num_tok,768], bbox)
        self.enabled = True

    def set_active(self, items):
        self.active = items


class EntityIPAttnProcessor(torch.nn.Module):
    """attn2 processor: standard text cross-attn + per-entity bbox-localized
    IP image cross-attn, using TRAINED to_k_ip/to_v_ip weights."""
    def __init__(self, controller, to_k_ip, to_v_ip):
        super().__init__()
        self.ctrl = controller
        self.to_k_ip = to_k_ip      # nn.Linear(768, inner)
        self.to_v_ip = to_v_ip

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, Hs, Ws = hidden_states.shape
            hidden_states = hidden_states.view(B, C, Hs*Ws).transpose(1, 2)
        Bsz, S, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        q = attn.to_q(hidden_states)
        enc = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        if attn.norm_cross:
            enc = attn.norm_encoder_hidden_states(enc)
        k = attn.to_k(enc); v = attn.to_v(enc)
        inner = k.shape[-1]; nh = attn.heads; hd = inner // nh
        q4 = q.view(Bsz, -1, nh, hd).transpose(1, 2)
        k4 = k.view(Bsz, -1, nh, hd).transpose(1, 2)
        v4 = v.view(Bsz, -1, nh, hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=None,
                                             dropout_p=0., is_causal=False)
        out = out.transpose(1, 2).reshape(Bsz, S, inner)   # (B,S,inner) text attn

        # ── per-entity bbox-localized IP image cross-attention ──────────────
        gate = (self.ctrl.cur_t is None) or (self.ctrl.cur_t < self.ctrl.t_apply_below)
        if (self.ctrl.enabled and self.ctrl.active and gate and S == int(S**0.5)**2):
            Hf = Wf = int(S**0.5)
            cond_lo = Bsz // 2 if (self.ctrl.cfg and Bsz % 2 == 0) else 0
            dev = hidden_states.device
            qc = q4[cond_lo:cond_lo+1]                      # (1,nh,S,hd) cond query
            add = torch.zeros(1, S, inner, device=dev, dtype=out.dtype)
            for name, tokens, bbox in self.ctrl.active:
                m = bbox_mask_feathered(bbox, Hf, Wf, dev, feather=self.ctrl.feather)
                if float(m.max()) == 0:
                    continue
                t = tokens.to(self.to_k_ip.weight.dtype).unsqueeze(0)  # (1,num_tok,768)
                k_ip = self.to_k_ip(t).to(qc.dtype).view(1, -1, nh, hd).transpose(1, 2)
                v_ip = self.to_v_ip(t).to(qc.dtype).view(1, -1, nh, hd).transpose(1, 2)
                ip = F.scaled_dot_product_attention(qc, k_ip, v_ip, dropout_p=0.)
                ip = ip.transpose(1, 2).reshape(1, S, inner)
                add = add + ip * m.view(1, S, 1)            # soft bbox-localized
            out[cond_lo:cond_lo+1] = out[cond_lo:cond_lo+1] + self.ctrl.scale * add

        out = out.to(q.dtype)
        out = attn.to_out[0](out); out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(Bsz, C, Hs, Ws)
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out


def extract_ip_adapter(ipa_pipe):
    """From a StableDiffusionPipeline that has load_ip_adapter'd weights, pull:
       - per-attn2-layer (to_k_ip, to_v_ip)
       - the image projection layer (CLIP embed -> image tokens)
       - the CLIP image encoder + feature extractor
    """
    layers = {}
    for name, proc in ipa_pipe.unet.attn_processors.items():
        if hasattr(proc, "to_k_ip"):
            layers[name] = (proc.to_k_ip[0], proc.to_v_ip[0])   # single ip-adapter
    image_proj = ipa_pipe.unet.encoder_hid_proj.image_projection_layers[0]
    # IP-Adapter-Plus uses a Resampler over CLIP PATCH features (penultimate
    # hidden states), not the pooled image_embeds → richer/stronger identity.
    plus = "Plus" in type(image_proj).__name__
    return dict(layers=layers, image_proj=image_proj, plus=plus,
                image_encoder=ipa_pipe.image_encoder,
                feature_extractor=ipa_pipe.feature_extractor)


@torch.no_grad()
def entity_tokens(ip, pil_image, device, obj_mask=None):
    """CLIP(ref) -> image_proj -> (num_tokens, 768) entity image tokens.
    Plus: feed penultimate patch hidden states; base: pooled image_embeds.

    obj_mask (H,W in [0,1]): if given (Plus only), zero out BACKGROUND patch
    tokens before the Resampler so the identity tokens encode the object only —
    no anchor background leaks into the generated scene."""
    import torch as _t
    fe = ip["feature_extractor"]; enc = ip["image_encoder"]; proj = ip["image_proj"]
    px = fe(pil_image, return_tensors="pt").pixel_values.to(device, enc.dtype)
    if ip.get("plus"):
        feats = enc(px, output_hidden_states=True).hidden_states[-2]  # (1,1+P,1280)
        if obj_mask is not None:
            P = feats.shape[1] - 1                     # patch count (e.g. 256)
            g = int(P ** 0.5)
            m = _t.tensor(obj_mask, dtype=_t.float32, device=device)[None, None]
            m = _t.nn.functional.interpolate(m, size=(g, g), mode="area")
            keep = (m.reshape(-1) > 0.4).to(feats.dtype)            # (P,) patch keep
            patch = feats[:, 1:, :] * keep[None, :, None]           # zero bg patches
            feats = _t.cat([feats[:, :1, :], patch], dim=1)         # keep CLS
        toks = proj(feats)                            # (1,16,768)
    else:
        emb = enc(px).image_embeds
        toks = proj(emb)
    return toks.squeeze(0)


def install_entity_ip(unet, controller, ip):
    """Replace attn2 processors with EntityIPAttnProcessor (trained ip weights),
    keep attn1 untouched."""
    procs = {}
    n = 0
    for name in list(unet.attn_processors.keys()):
        mod = name[:-len(".processor")] if name.endswith(".processor") else name
        if mod.endswith("attn2") and name in ip["layers"]:
            k_ip, v_ip = ip["layers"][name]
            procs[name] = EntityIPAttnProcessor(controller, k_ip, v_ip)
            n += 1
        else:
            procs[name] = unet.attn_processors[name]
    unet.set_attn_processor(procs)
    # forward pre-hook to track the current timestep for the controller's gate
    def _hook(module, args, kwargs):
        t = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        if t is not None:
            controller.cur_t = float(t.flatten()[0].item() if torch.is_tensor(t) else t)
        return None
    unet.register_forward_pre_hook(_hook, with_kwargs=True)
    print(f"[entity-ip] installed on {n} attn2 layers")
