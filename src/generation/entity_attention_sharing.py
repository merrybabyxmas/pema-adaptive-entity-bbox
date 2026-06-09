"""
v13: Training-free BBox-Localized Same-Entity Attention Sharing.

Idea (no training, no loss): reuse a subject's diffusion self-attention
representation across shots, but ONLY:
  - for entities present in the shot (presence-aware),
  - inside that entity's predicted bbox (spatially localized),
  - from the entity's own anchor bank (first occurrence).

Mechanism (per self-attn layer, per denoising timestep t):
  O_self = Attn(Q, K_cur, V_cur)                              # normal self-attn
  O_ref  = Attn(Q_bbox, K_anchor[e,layer,t], V_anchor[...])   # cross to anchor
  O[bbox_e] = (1-alpha(t)) * O_self[bbox_e] + alpha(t) * O_ref[bbox_e]
  O[outside] = O_self

Anchor bank is filled during an entity's FIRST-occurrence shot (capture mode)
and read in later shots (inject mode). Banks are keyed by timestep value, so the
same scheduler/steps align capture↔inject automatically.

Only standard SD/GLIGEN attn1 (self-attention) processors are replaced.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def bbox_token_mask(bbox, H, W, device, pad=0.06):
    """Return bool mask (H*W,) for latent tokens inside (padded) bbox [x1,y1,x2,y2]."""
    x1, y1, x2, y2 = bbox
    x1 = max(0., x1 - pad); y1 = max(0., y1 - pad)
    x2 = min(1., x2 + pad); y2 = min(1., y2 + pad)
    ys = (torch.arange(H, device=device).float() + 0.5) / H
    xs = (torch.arange(W, device=device).float() + 0.5) / W
    m = ((ys >= y1) & (ys < y2)).view(H, 1) & ((xs >= x1) & (xs < x2)).view(1, W)
    return m.view(H * W)


class SharingController:
    """Holds per-step shared state driving all SharedSelfAttnProcessors."""
    def __init__(self, share_layers, alpha_schedule=None, cfg=True, alpha_max=0.55):
        # set of (dotted) module-name substrings where sharing is active
        self.share_layers = share_layers
        self.cfg = cfg
        self.alpha_max = alpha_max   # late-stage sharing strength (knob → consistency)
        self.mode = "off"            # 'off' | 'on'
        self.cur_t = None            # current timestep (int key)
        self.active = []             # list of (entity_name, bbox)
        self.bank = {}               # bank[entity][layer_key][t_int] = (K_bbox, V_bbox)
        self.freeze_bank = False     # if True, never capture (anchor already built)
        self.alpha_schedule = alpha_schedule or self._default_alpha

    def _default_alpha(self, t):
        # t in [0,1000]; share weakly at high noise, strongly late. Scaled by alpha_max.
        a = self.alpha_max
        if t > 700:   return a * 0.18
        if t > 300:   return a * 0.55
        return a

    def reset_active(self):
        self.active = []

    def capture_entity(self, layer_key, entity, K, V):
        self.bank.setdefault(entity, {}).setdefault(layer_key, {})[int(self.cur_t)] = (K, V)

    def get_anchor(self, layer_key, entity):
        return self.bank.get(entity, {}).get(layer_key, {}).get(int(self.cur_t))


class SharedSelfAttnProcessor:
    """AttnProcessor2_0 + capture/inject of bbox-localized same-entity K/V."""
    def __init__(self, controller: SharingController, layer_key: str, active_share: bool):
        self.ctrl = controller
        self.layer_key = layer_key
        self.active_share = active_share   # whether this layer participates

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, Hs, Ws = hidden_states.shape
            hidden_states = hidden_states.view(B, C, Hs * Ws).transpose(1, 2)
        Bsz, S, _ = hidden_states.shape
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        q = attn.to_q(hidden_states)
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        inner = k.shape[-1]; nh = attn.heads; hd = inner // nh
        q4 = q.view(Bsz, -1, nh, hd).transpose(1, 2)
        k4 = k.view(Bsz, -1, nh, hd).transpose(1, 2)
        v4 = v.view(Bsz, -1, nh, hd).transpose(1, 2)

        out = F.scaled_dot_product_attention(q4, k4, v4, dropout_p=0., is_causal=False)

        do_share = (self.active_share and self.ctrl.mode == "on"
                    and self.ctrl.active and S == int(S**0.5)**2)
        if do_share:
            Hf = Wf = int(S ** 0.5)
            # CFG: rows [0:B] uncond, [B:2B] cond. Operate on the cond half.
            cond_lo = Bsz // 2 if self.ctrl.cfg and Bsz % 2 == 0 else 0
            dev = hidden_states.device
            for entity, bbox in self.ctrl.active:
                m = bbox_token_mask(bbox, Hf, Wf, dev)          # (S,)
                if m.sum() == 0:
                    continue
                anc = self.ctrl.get_anchor(self.layer_key, entity)
                if anc is None:
                    if self.ctrl.freeze_bank:
                        continue                       # anchor pre-built; no capture
                    # first occurrence → CAPTURE this entity's bbox K/V
                    Kb = k4[cond_lo:cond_lo+1, :, m, :].detach()
                    Vb = v4[cond_lo:cond_lo+1, :, m, :].detach()
                    self.ctrl.capture_entity(self.layer_key, entity, Kb, Vb)
                else:
                    # seen before → INJECT anchor into current bbox tokens
                    Kb, Vb = anc
                    alpha = self.ctrl.alpha_schedule(self.ctrl.cur_t)
                    qsel = q4[cond_lo:cond_lo+1, :, m, :]        # (1,nh,nb,hd)
                    o_ref = F.scaled_dot_product_attention(qsel, Kb, Vb, dropout_p=0.)
                    o_cond = out[cond_lo:cond_lo+1]
                    o_cond[:, :, m, :] = (1 - alpha) * o_cond[:, :, m, :] + alpha * o_ref
                    out[cond_lo:cond_lo+1] = o_cond

        out = out.transpose(1, 2).reshape(Bsz, S, inner).to(q.dtype)
        out = attn.to_out[0](out); out = attn.to_out[1](out)
        if input_ndim == 4:
            out = out.transpose(-1, -2).reshape(Bsz, C, Hs, Ws)
        if attn.residual_connection:
            out = out + residual
        out = out / attn.rescale_output_factor
        return out


def install_sharing(unet, controller: SharingController):
    """Replace attn1 (self-attn) processors with SharedSelfAttnProcessor.
    attn2 (cross-attn) processors are left untouched."""
    procs = {}
    n = 0
    for name in list(unet.attn_processors.keys()):
        mod_name = name[:-len(".processor")] if name.endswith(".processor") else name
        if mod_name.endswith("attn1"):
            active = any(s in mod_name for s in controller.share_layers)
            n += int(active)
            procs[name] = SharedSelfAttnProcessor(controller, mod_name, active)
        else:
            procs[name] = unet.attn_processors[name]
    unet.set_attn_processor(procs)
    print(f"[sharing] active in {n} attn1 layers (share_layers={controller.share_layers})")


def attach_timestep_hook(unet, controller: SharingController):
    """Forward pre-hook to read the current timestep into the controller."""
    def hook(module, args, kwargs):
        t = kwargs.get("timestep", args[1] if len(args) > 1 else None)
        if t is not None:
            controller.cur_t = float(t.flatten()[0].item() if torch.is_tensor(t) else t)
        return None
    return unet.register_forward_pre_hook(hook, with_kwargs=True)
