"""
Phase 4: Train EntityStyleAdapter with diffusion noise prediction loss.

Objective:
  Given (noisy_latent z_t, timestep t, text, entity_refs, style_ref, bboxes),
  predict the noise ε that was added to the clean latent z_0.

  L = E[||ε - ε_θ(z_t, t, c)||²]
    c = (text_emb, entity_tokens, style_tokens, entity_bboxes)

Training:
  - Freeze entire SD UNet (only adapter K/V projections + gamma trained)
  - Freeze VAE (encode images to latents, decode for logging only)
  - Freeze text encoder and CLIP image encoder
  - Load existing StyleEncoder (Phase 2) for style token extraction

Data (from generate_phase4_data.py):
  data/phase4_train/
    samples/sample_XXXX/
      scene.png           ← target image (encode to latent z_0)
      style_bg.png        ← background for style ref
      entity_NAME.png     ← entity reference crop
      metadata.json       ← prompt, bboxes, entity names

Loss breakdown:
  L_total = L_noise                            (primary)
          + λ_ent  * L_entity_contrastive      (optional, reg)
  L_entity_contrastive: entity_to_k/v outputs for same entity should cluster

Output: outputs/runs/phase4_adapter/adapter_best.pt

Usage:
  python scripts/train_phase4.py \
    --data-dir data/phase4_train \
    --style-encoder-path outputs/runs/pema_style_encoder/style_encoder_best.pt \
    --epochs 50 --batch-size 4
"""
import sys, os, json, random, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import torchvision.transforms as T

from diffusers import (
    StableDiffusionGLIGENPipeline, DDPMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTokenizer, CLIPTextModel

from src.memory.entity_encoder import EntityEncoder
from src.model.style_encoder import StyleEncoder
from src.model.entity_style_adapter import EntityStyleAdapter
from src.generation.pema_pipeline import GLIGEN_MODEL
from src.utils.logging import get_logger

logger = get_logger("train_phase4")


# ── Dataset ───────────────────────────────────────────────────────────────────

class Phase4Dataset(Dataset):
    """
    Each sample: (scene_latent, text_emb, entity_tokens, style_tokens, entity_bboxes)

    All heavy encoding (VAE, CLIP, StyleEncoder) is pre-cached at __init__
    so training steps are fast (no encoder overhead).
    """

    def __init__(
        self,
        data_dir: str,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        entity_encoder: EntityEncoder,
        style_encoder: StyleEncoder,
        device: str,
        vae_scale: float = 0.18215,
        image_size: int = 512,
        normalize_tokens: bool = True,
        entity_mode: str = "pooled",
        entity_grid: int = 4,
    ):
        self.normalize_tokens = normalize_tokens
        self.entity_mode = entity_mode      # "pooled" (1024) | "patch" (grid^2 x 1280)
        self.entity_grid = entity_grid
        self.vae_scale = vae_scale
        self.data_dir  = Path(data_dir)
        index_path     = self.data_dir / "scene_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"scene_index.json not found in {data_dir}. "
                "Run scripts/generate_phase4_data.py first."
            )
        sample_dirs = json.loads(index_path.read_text())
        logger.info(f"Phase4Dataset: {len(sample_dirs)} samples found")

        to_tensor  = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
        normalize  = T.Normalize([0.5], [0.5])

        self.samples = []

        with torch.no_grad():
            for i, rel_dir in enumerate(sample_dirs):
                sdir = self.data_dir / rel_dir
                meta = json.loads((sdir / "metadata.json").read_text())

                # ── Scene image → VAE latent ───────────────────────────────
                scene = Image.open(str(sdir / "scene.png")).convert("RGB")
                img_t = normalize(to_tensor(scene)).unsqueeze(0).to(device).half()
                z0    = vae.encode(img_t).latent_dist.sample() * vae_scale
                z0    = z0.squeeze(0).float().cpu()   # (4, 64, 64)

                # ── Text embedding ─────────────────────────────────────────
                prompt = meta["prompt"]
                tok    = tokenizer(
                    prompt, padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt"
                ).to(device)
                text_emb = text_encoder(**tok).last_hidden_state.squeeze(0).float().cpu()
                # (77, 768)

                # ── Entity tokens (CLIP of cross-image ref) ───────────────
                # ref_image is either:
                #   (a) absolute path  — new cross-image format (pema_train ref)
                #   (b) filename       — old format (sdir-relative crop)
                entity_clips = []
                bboxes       = []
                for e in meta["entities"]:
                    ref_str  = e["ref_image"]
                    ref_path = Path(ref_str)
                    if not ref_path.is_absolute() or not ref_path.exists():
                        ref_path = sdir / ref_str        # fallback: sdir-relative
                    if not ref_path.exists():
                        ref_img = _crop_box(scene, e["box_xyxy"])  # last resort
                    else:
                        ref_img = Image.open(str(ref_path)).convert("RGB")
                    if self.entity_mode == "patch":
                        clip_emb = entity_encoder.encode_patches(
                            ref_img, grid=self.entity_grid).float().cpu()  # (K_e,1280)
                    else:
                        clip_emb = entity_encoder.encode(ref_img).float().cpu()  # (1024,)
                    entity_clips.append(clip_emb)
                    bboxes.append(e["box_xyxy"])

                entity_tokens = torch.stack(entity_clips)  # (n_ent,1024) or (n_ent,K_e,1280)
                # L2-normalize entity tokens (last dim) so the conditioning
                # distribution is invariant to CLIP embedding magnitude — matches
                # the inference memory bank (fusion is re-normalized too).
                if self.normalize_tokens:
                    entity_tokens = entity_tokens / (
                        entity_tokens.norm(dim=-1, keepdim=True) + 1e-8)

                # ── Style tokens (StyleEncoder from background) ────────────
                bg_path = sdir / "style_bg.png"
                if bg_path.exists():
                    bg_img   = Image.open(str(bg_path)).convert("RGB")
                    bg_clip  = entity_encoder.encode(bg_img).to(device)
                    sty_toks = style_encoder(bg_clip.unsqueeze(0)).squeeze(0).float().cpu()
                    # (K_g, 768)
                else:
                    sty_toks = None

                self.samples.append({
                    "z0":            z0,                    # (4, 64, 64)
                    "text_emb":      text_emb,              # (77, 768)
                    "entity_tokens": entity_tokens,         # (n_ent, 1024)
                    "entity_bboxes": bboxes,                # [(x1,y1,x2,y2),...]
                    "style_tokens":  sty_toks,              # (K_g, 768) or None
                    "prompt":        prompt,
                })

                if (i + 1) % 50 == 0:
                    logger.info(f"  Pre-encoded {i+1}/{len(sample_dirs)}")

        logger.info(f"Pre-encoding done. {len(self.samples)} samples cached.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _crop_box(image: Image.Image, box: list[float]) -> Image.Image:
    W, H = image.size
    x1 = int(box[0] * W); y1 = int(box[1] * H)
    x2 = int(box[2] * W); y2 = int(box[3] * H)
    return image.crop((max(0,x1), max(0,y1), min(W,x2), min(H,y2)))


def collate_fn(batch):
    """Custom collate: entity_bboxes and style_tokens have variable shapes."""
    z0           = torch.stack([b["z0"]       for b in batch])
    text_emb     = torch.stack([b["text_emb"] for b in batch])
    entity_tokens = [b["entity_tokens"]        for b in batch]  # list of (n_ent_i, 1024)
    entity_bboxes = [b["entity_bboxes"]        for b in batch]  # list of lists
    style_tokens  = [b["style_tokens"]         for b in batch]  # list of (K_g, 768) or None
    return {
        "z0":            z0,
        "text_emb":      text_emb,
        "entity_tokens": entity_tokens,
        "entity_bboxes": entity_bboxes,
        "style_tokens":  style_tokens,
    }


# ── Validation helpers ────────────────────────────────────────────────────────

def build_val_samples(data_dir: Path, dataset: "Phase4Dataset", n_val: int = 6, seed: int = 0):
    """
    Pick n_val fixed samples from the dataset for per-epoch visual validation.
    Prefer multi-entity samples (better test of bbox-conditioned entity attention).
    Returns list of dicts with raw metadata needed for pipeline generation.
    """
    index = json.loads((data_dir / "scene_index.json").read_text())
    rng   = random.Random(seed)

    multi  = [i for i, s in enumerate(dataset.samples) if s["entity_tokens"].shape[0] > 1]
    single = [i for i, s in enumerate(dataset.samples) if s["entity_tokens"].shape[0] == 1]

    n_multi  = min(n_val * 2 // 3, len(multi))
    n_single = min(n_val - n_multi, len(single))
    chosen   = rng.sample(multi, n_multi) + rng.sample(single, n_single)
    rng.shuffle(chosen)

    val_samples = []
    for idx in chosen:
        s    = dataset.samples[idx]
        sdir = data_dir / index[idx]
        meta = json.loads((sdir / "metadata.json").read_text())

        val_samples.append({
            "idx":           idx,
            "prompt":        s["prompt"],
            "entity_tokens": s["entity_tokens"],   # (n_ent, 1024)  float32 cpu
            "entity_bboxes": s["entity_bboxes"],   # list of [x1,y1,x2,y2]
            "style_tokens":  s["style_tokens"],    # (K_g, 768) or None
            "entity_names":  [e["name"] for e in meta["entities"]],
            "scene_path":    str(sdir / "scene.png"),
            "ref_paths":     [e.get("ref_image", "") for e in meta["entities"]],
        })

    logger.info(f"Validation set: {len(val_samples)} samples "
                f"({n_multi} multi-entity, {n_single} single-entity)")
    return val_samples


def _draw_bboxes(img: Image.Image, bboxes: list, names: list) -> Image.Image:
    """Overlay bbox rectangles on image for sanity-check visualisation."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size
    colors = ["red", "blue", "green", "orange"]
    for i, (box, name) in enumerate(zip(bboxes, names)):
        x1, y1, x2, y2 = int(box[0]*W), int(box[1]*H), int(box[2]*W), int(box[3]*H)
        col = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=col, width=3)
        draw.text((x1 + 4, y1 + 4), name[:12], fill=col)
    return img


def validate_epoch(
    epoch: int,
    val_samples: list,
    adapter: "EntityStyleAdapter",
    pipe: "StableDiffusionGLIGENPipeline",
    device: torch.device,
    out_dir: Path,
    n_steps: int = 20,
):
    """
    Generate validation images for each val sample using adapter + GLIGEN.
    Saves per epoch:
      val_epoch_NNN/
        sample_NN_generated.png   ← generated with adapter
        sample_NN_target.png      ← ground-truth scene
        sample_NN_ref_entK.png    ← entity reference used
        sample_NN_annotated.png   ← generated with bbox overlay
    """
    val_dir = out_dir / f"val_epoch_{epoch:03d}"
    val_dir.mkdir(parents=True, exist_ok=True)

    unet = pipe.unet
    unet.eval()

    with torch.no_grad():
        for j, vs in enumerate(val_samples):
            ent_toks = vs["entity_tokens"].unsqueeze(0).to(device).float()
            sty_toks = vs["style_tokens"]
            if sty_toks is not None:
                sty_toks = sty_toks.unsqueeze(0).to(device).float()

            adapter.set_conditions(ent_toks, vs["entity_bboxes"], sty_toks)

            try:
                result = pipe(
                    prompt=vs["prompt"],
                    gligen_phrases=vs["entity_names"],
                    gligen_boxes=vs["entity_bboxes"],
                    gligen_scheduled_sampling_beta=1.0,
                    num_inference_steps=n_steps,
                    height=512, width=512,
                    generator=torch.Generator(str(device)).manual_seed(j),
                )
                gen_img = result.images[0]
            except Exception as exc:
                logger.warning(f"  val sample {j} generation failed: {exc}")
                adapter.clear_conditions()
                continue
            finally:
                adapter.clear_conditions()

            # Save generated image
            gen_img.save(str(val_dir / f"sample_{j:02d}_generated.png"))

            # Annotated with bboxes
            ann = _draw_bboxes(gen_img, vs["entity_bboxes"], vs["entity_names"])
            ann.save(str(val_dir / f"sample_{j:02d}_annotated.png"))

            # Ground-truth scene
            if Path(vs["scene_path"]).exists():
                shutil.copy(vs["scene_path"], str(val_dir / f"sample_{j:02d}_target.png"))

            # Entity reference images
            for k, ref_str in enumerate(vs["ref_paths"]):
                rp = Path(ref_str)
                if not rp.is_absolute():
                    rp = Path(vs["scene_path"]).parent / ref_str
                if rp.exists():
                    shutil.copy(str(rp), str(val_dir / f"sample_{j:02d}_ref_ent{k}.png"))

    unet.train()
    logger.info(f"  → Val images saved: {val_dir}")


# ── Training ──────────────────────────────────────────────────────────────────

def train_step(
    adapter: EntityStyleAdapter,
    unet: nn.Module,
    scheduler: DDPMScheduler,
    batch: dict,
    device: str,
    dtype: torch.dtype,
    entity_dropout: float = 0.0,
):
    """
    Single diffusion training step.
    Processes each sample independently (variable entity counts).

    entity_dropout: probability of zeroing out entity tokens (conditional dropout).
    Forces K/V projections to be robust; enables CFG-style inference.
    """
    z0       = batch["z0"].to(device).float()
    text_emb = batch["text_emb"].to(device)
    B        = z0.shape[0]

    t       = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
    noise   = torch.randn_like(z0)
    z_noisy = scheduler.add_noise(z0, noise, t)

    loss_sum = torch.tensor(0., device=device)

    for i in range(B):
        ent_toks   = batch["entity_tokens"][i].unsqueeze(0).to(device).float()
        ent_bboxes = batch["entity_bboxes"][i]
        sty_toks   = None
        if batch["style_tokens"][i] is not None:
            sty_toks = batch["style_tokens"][i].unsqueeze(0).to(device).float()

        # Conditional dropout: randomly zero entity / style tokens independently
        if entity_dropout > 0.0:
            if random.random() < entity_dropout:
                ent_toks = torch.zeros_like(ent_toks)
            if sty_toks is not None and random.random() < entity_dropout:
                sty_toks = torch.zeros_like(sty_toks)

        adapter.set_conditions(ent_toks, ent_bboxes, sty_toks)

        with torch.amp.autocast("cuda", dtype=dtype):
            noise_pred = unet(
                z_noisy[i:i+1],
                t[i:i+1],
                encoder_hidden_states=text_emb[i:i+1],
            ).sample

        loss_i = F.mse_loss(noise_pred.float(), noise[i:i+1])
        loss_sum = loss_sum + loss_i

    adapter.clear_conditions()
    return loss_sum / B


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     default="data/phase4_train")
    parser.add_argument("--out-dir",      default="outputs/runs/phase4_adapter")
    parser.add_argument("--gligen-model", default=GLIGEN_MODEL)
    parser.add_argument("--style-encoder-path",
                        default="outputs/runs/pema_style_encoder/style_encoder_best.pt")
    parser.add_argument("--n-style-tokens", type=int, default=4)

    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--batch-size",  type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--lr-warmup",   type=int,   default=100,
                        help="Warmup steps for LR scheduler")
    parser.add_argument("--gamma-entity",type=float, default=0.1)
    parser.add_argument("--gamma-style", type=float, default=0.05)
    parser.add_argument("--grad-clip",   type=float, default=1.0)
    parser.add_argument("--log-every",   type=int,   default=20)
    parser.add_argument("--save-every",  type=int,   default=10)
    parser.add_argument("--val-samples", type=int,   default=6,
                        help="Number of fixed samples for per-epoch validation generation")
    parser.add_argument("--val-steps",   type=int,   default=20,
                        help="Denoising steps for validation generation (lower = faster)")
    parser.add_argument("--no-val",      action="store_true",
                        help="Skip per-epoch validation generation")
    parser.add_argument("--fix-gamma",   action="store_true",
                        help="Freeze gamma params (no gradient). Trains K/V only.")
    parser.add_argument("--lr-kv",       type=float, default=1e-5,
                        help="Learning rate for K/V projection params")
    parser.add_argument("--entity-dropout", type=float, default=0.0,
                        help="Probability to zero entity/style tokens during training")
    parser.add_argument("--style-layers", nargs="*", default=None,
                        help="Restrict style branch to UNet blocks whose name "
                             "contains these substrings (InstantStyle). "
                             "e.g. --style-layers up_blocks.1  | "
                             "--style-layers down_blocks.2 mid_block up_blocks.1. "
                             "Omit for all-layers (legacy).")
    parser.add_argument("--entity-mode", choices=["pooled", "patch"], default="pooled",
                        help="pooled=1 CLIP vector/entity (1024d); "
                             "patch=grid^2 CLIP patch tokens/entity (1280d, "
                             "IP-Adapter-Plus style, richer identity).")
    parser.add_argument("--entity-grid", type=int, default=4,
                        help="grid size for patch mode (entity tokens = grid^2)")
    parser.add_argument("--no-normalize-tokens", action="store_true",
                        help="Disable L2-normalization of entity tokens "
                             "(legacy; v8+ normalizes by default to match the "
                             "inference memory-bank fusion distribution).")
    parser.add_argument("--entity-layers", nargs="*", default=None,
                        help="Restrict entity branch to UNet blocks whose name "
                             "contains these substrings. e.g. exclude 64-res "
                             "detail layers: --entity-layers down_blocks.1 "
                             "down_blocks.2 mid_block up_blocks.1 up_blocks.2. "
                             "Omit for all-layers.")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    base    = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)
    dtype   = torch.float16

    # ── Load frozen GLIGEN pipeline ───────────────────────────────────────────
    logger.info(f"Loading GLIGEN pipeline: {args.gligen_model}")
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        args.gligen_model, torch_dtype=dtype,
    ).to(str(device))
    pipe.set_progress_bar_config(disable=True)

    unet         = pipe.unet
    vae          = pipe.vae
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer

    # Freeze everything
    for m in [unet, vae, text_encoder]:
        for p in m.parameters():
            p.requires_grad_(False)

    # ── DDPM noise scheduler ──────────────────────────────────────────────────
    scheduler = DDPMScheduler.from_pretrained(
        args.gligen_model, subfolder="scheduler"
    )

    # ── Entity + style encoders (frozen) ─────────────────────────────────────
    logger.info("Loading EntityEncoder (CLIP ViT-H/14)...")
    entity_encoder = EntityEncoder(device=str(device))

    logger.info("Loading StyleEncoder...")
    style_enc_path = base / args.style_encoder_path
    if style_enc_path.exists():
        ckpt = torch.load(str(style_enc_path), map_location=device, weights_only=False)
        n_tok = ckpt.get("n_tokens", args.n_style_tokens)
        style_encoder = StyleEncoder(n_tokens=n_tok).to(device)
        style_encoder.load_state_dict(ckpt["model"])
        style_encoder.eval()
        for p in style_encoder.parameters(): p.requires_grad_(False)
        logger.info(f"StyleEncoder loaded (K_g={n_tok})")
    else:
        logger.warning("StyleEncoder checkpoint not found — using random init")
        style_encoder = StyleEncoder(n_tokens=args.n_style_tokens).to(device)
        style_encoder.eval()
        for p in style_encoder.parameters(): p.requires_grad_(False)

    # ── Build EntityStyleAdapter ──────────────────────────────────────────────
    logger.info("Building EntityStyleAdapter...")
    entity_dim = 1280 if args.entity_mode == "patch" else 1024
    adapter = EntityStyleAdapter(
        unet,
        entity_dim=entity_dim,
        gamma_entity_init=args.gamma_entity,
        gamma_style_init=args.gamma_style,
        style_layers=args.style_layers,
        entity_layers=args.entity_layers,
    )
    adapter.register_to_unet()

    param_counts = adapter.parameter_count()
    logger.info(
        f"Adapter params: {param_counts['total']/1e6:.2f}M total "
        f"({param_counts['entity_kv']/1e6:.2f}M entity_kv, "
        f"{param_counts['style_kv']/1e6:.2f}M style_kv)"
    )

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    logger.info("Pre-encoding training data...")
    dataset = Phase4Dataset(
        data_dir=str(base / args.data_dir),
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        entity_encoder=entity_encoder,
        style_encoder=style_encoder,
        device=str(device),
        vae_scale=vae.config.get("scaling_factor", 0.18215),
        normalize_tokens=not args.no_normalize_tokens,
        entity_mode=args.entity_mode,
        entity_grid=args.entity_grid,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    logger.info(f"DataLoader: {len(dataset)} samples, batch_size={args.batch_size}")

    # ── Validation set ────────────────────────────────────────────────────────
    val_samples = []
    if not args.no_val:
        val_samples = build_val_samples(
            base / args.data_dir, dataset,
            n_val=args.val_samples, seed=42,
        )

    # ── Optimizer + LR scheduler ──────────────────────────────────────────────
    # Separate K/V projection params from gamma params
    kv_params, gamma_params = [], []
    for proc in adapter.processors.values():
        for name, param in proc.named_parameters():
            if 'gamma' in name:
                gamma_params.append(param)
            else:
                kv_params.append(param)

    if args.fix_gamma:
        for p in gamma_params:
            p.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            kv_params, lr=args.lr_kv, betas=(0.9, 0.999), weight_decay=1e-4,
        )
        logger.info(f"fix-gamma ON: gamma frozen, K/V lr={args.lr_kv:.1e}")
    else:
        optimizer = torch.optim.AdamW([
            {"params": kv_params,    "lr": args.lr_kv},
            {"params": gamma_params, "lr": args.lr},
        ], betas=(0.9, 0.999), weight_decay=1e-4)
        logger.info(f"K/V lr={args.lr_kv:.1e}, gamma lr={args.lr:.1e}")

    total_steps = args.epochs * len(loader)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_sched = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.lr_warmup
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - args.lr_warmup)
    )
    lr_sched = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[args.lr_warmup]
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    global_step = 0
    unet.train()
    adapter.processors.to(device)

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.
        n_batches  = 0

        for batch in loader:
            loss = train_step(
                adapter, unet, scheduler, batch, str(device), dtype,
                entity_dropout=args.entity_dropout,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in adapter.trainable_parameters() if p.requires_grad],
                args.grad_clip,
            )
            optimizer.step()
            lr_sched.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if global_step % args.log_every == 0:
                # Log current gamma values
                g_e = [proc.gamma_entity.item()
                       for proc in adapter.processors.values()
                       if hasattr(proc, "gamma_entity")]
                g_s = [proc.gamma_style.item()
                       for proc in adapter.processors.values()
                       if hasattr(proc, "gamma_style")]
                mean_ge = sum(g_e) / len(g_e) if g_e else 0
                mean_gs = sum(g_s) / len(g_s) if g_s else 0
                logger.info(
                    f"Step {global_step:5d} | loss={loss.item():.4f} "
                    f"| γ_e={mean_ge:.4f} γ_g={mean_gs:.4f} "
                    f"| lr={lr_sched.get_last_lr()[0]:.2e}"
                )

        avg = epoch_loss / n_batches
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | avg_loss={avg:.4f}")

        if avg < best_loss:
            best_loss = avg
            adapter.save(str(out_dir / "adapter_best.pt"), epoch=epoch, loss=avg)
            logger.info(f"  → New best! Saved adapter_best.pt")

        if epoch % args.save_every == 0:
            adapter.save(
                str(out_dir / f"adapter_epoch{epoch:04d}.pt"),
                epoch=epoch, loss=avg,
            )

        # Per-epoch validation generation
        if val_samples:
            logger.info(f"  Generating validation samples (epoch {epoch})...")
            validate_epoch(
                epoch=epoch,
                val_samples=val_samples,
                adapter=adapter,
                pipe=pipe,
                device=device,
                out_dir=out_dir,
                n_steps=args.val_steps,
            )

    logger.info(f"\nTraining done. Best loss={best_loss:.4f}")
    logger.info(f"Saved: {out_dir}/adapter_best.pt")


if __name__ == "__main__":
    main()
