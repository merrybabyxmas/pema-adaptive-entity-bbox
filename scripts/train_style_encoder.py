"""
Phase 2: Train StyleEncoder with SimCLR-style consistency on background crops.

Objective: style tokens must be consistent under spatial augmentation but
different across scenes — capturing color palette, lighting, texture.

Augmentation strategy:
  - Two random spatial crops of same image → positive pair (same style)
  - Crops from different images → negatives
  - Color jitter intentionally EXCLUDED so style tokens retain color info
  - Only spatial/geometric augmentation: crop, flip, mild blur

Loss: InfoNCE on mean-pooled style tokens (aggregate over K_g)

Data: entity images from data/pema_train/images/ (pre-CLIP-encoded)
Background masking: entity bbox regions are blurred before CLIP encoding
so style tokens focus on scene style rather than entity-specific features.

Output: outputs/runs/pema_style_encoder/style_encoder_best.pt
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from src.memory.entity_encoder import EntityEncoder
from src.model.style_encoder import StyleEncoder, style_consistency_loss
from src.utils.logging import get_logger

logger = get_logger("train_style")


# ── Background extraction ─────────────────────────────────────────────────────

def mask_entities(image: Image.Image, entity_bboxes: list,
                  blur_radius: int = 40) -> Image.Image:
    """
    Replace entity bbox regions with heavy blur to isolate background style.
    entity_bboxes: list of [x1,y1,x2,y2] in [0,1] normalized coords.
    """
    from PIL import ImageDraw
    W, H = image.size
    blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for box in entity_bboxes:
        pad = int(0.05 * min(W, H))
        x1 = max(0, int(box[0] * W) - pad)
        y1 = max(0, int(box[1] * H) - pad)
        x2 = min(W, int(box[2] * W) + pad)
        y2 = min(H, int(box[3] * H) + pad)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    return Image.composite(blurred, image, mask)


# ── Augmentation (spatial only — preserve color/style) ───────────────────────

def spatial_augment(image: Image.Image, size: int = 224) -> Image.Image:
    """Random spatial crop + optional flip. No color jitter."""
    W, H = image.size
    crop_scale = random.uniform(0.6, 1.0)
    cw, ch = int(W * crop_scale), int(H * crop_scale)
    x0 = random.randint(0, W - cw)
    y0 = random.randint(0, H - ch)
    img = image.crop((x0, y0, x0 + cw, y0 + ch))
    if random.random() > 0.5:
        img = TF.hflip(img)
    img = img.resize((size, size), Image.LANCZOS)
    return img


# ── Dataset ───────────────────────────────────────────────────────────────────

class StylePairDataset(Dataset):
    """
    Returns (clip_emb_a, clip_emb_b, scene_id) where a and b are two
    spatially augmented views of the same background-masked entity image.
    Both views encoded with CLIP ViT-H/14 — cached in memory.
    """

    def __init__(self, img_dir: str, entity_list: list,
                 img_encoder: EntityEncoder, device: str,
                 n_augments: int = 8, clip_size: int = 224):
        import re
        def slug(name): return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())

        img_root = Path(img_dir)
        self.n_aug = n_augments
        self.img_encoder = img_encoder
        self.device = device

        # Pre-load images (PIL, not encoded) for on-the-fly augmentation
        self.items = []   # (scene_id, PIL image)
        scene_id = 0
        for entity in entity_list:
            for p in sorted((img_root / slug(entity)).glob("*.png")):
                img = Image.open(str(p)).convert("RGB")
                self.items.append((scene_id, img))
                scene_id += 1

        logger.info(
            f"StylePairDataset: {len(self.items)} scenes, "
            f"{n_augments} augmented views cached per scene"
        )

        # Pre-encode all augmented views to avoid repeated CLIP calls
        logger.info("Pre-encoding augmented views...")
        self.emb_cache: list[list[torch.Tensor]] = []
        for sid, img in self.items:
            views = []
            for _ in range(n_augments):
                aug = spatial_augment(img, clip_size)
                emb = img_encoder.encode(aug).cpu()   # (1024,)
                views.append(emb)
            self.emb_cache.append(views)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        scene_id, _ = self.items[idx]
        views = self.emb_cache[idx]
        a_idx, b_idx = random.sample(range(len(views)), 2)
        return {
            "emb_a":    views[a_idx],
            "emb_b":    views[b_idx],
            "scene_id": torch.tensor(scene_id, dtype=torch.long),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/pema_train")
    parser.add_argument("--out-dir",    default="outputs/runs/pema_style_encoder")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--temperature",type=float, default=0.1,
                        help="InfoNCE temperature for style (higher than entity, 0.07-0.15)")
    parser.add_argument("--n-tokens",   type=int,   default=4,
                        help="K_g: number of style tokens per image")
    parser.add_argument("--n-augments", type=int,   default=8,
                        help="Augmented views per image pre-cached at startup")
    parser.add_argument("--device",     default="cuda")
    args = parser.parse_args()

    base    = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    # ── Load CLIP encoder ─────────────────────────────────────────────────
    logger.info("Loading EntityEncoder (CLIP ViT-H/14)...")
    img_encoder = EntityEncoder(device=str(device))

    # ── Dataset ───────────────────────────────────────────────────────────
    data_dir    = base / args.data_dir
    entity_list = (data_dir / "entity_list.txt").read_text().splitlines()
    dataset = StylePairDataset(
        img_dir=str(data_dir / "images"),
        entity_list=entity_list,
        img_encoder=img_encoder,
        device=str(device),
        n_augments=args.n_augments,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    style_enc = StyleEncoder(n_tokens=args.n_tokens).to(device)
    n_params  = sum(p.numel() for p in style_enc.parameters())
    logger.info(f"StyleEncoder params: {n_params/1e6:.2f}M  (K_g={args.n_tokens})")

    optimizer = torch.optim.AdamW(style_enc.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        style_enc.train()
        total_loss = 0.

        for batch in loader:
            emb_a    = batch["emb_a"].to(device)        # (B, 1024)
            emb_b    = batch["emb_b"].to(device)        # (B, 1024)
            scene_id = batch["scene_id"].to(device)     # (B,)

            tokens_a = style_enc(emb_a)                 # (B, K_g, 768)
            tokens_b = style_enc(emb_b)                 # (B, K_g, 768)

            # Aggregate K_g tokens → single embedding for InfoNCE
            s_a = style_enc.aggregate(tokens_a)         # (B, 768)
            s_b = style_enc.aggregate(tokens_b)         # (B, 768)

            loss = style_consistency_loss(s_a, s_b, scene_id, args.temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(style_enc.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        logger.info(
            f"Epoch {epoch:4d}/{args.epochs} | loss={avg:.4f} "
            f"| lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch":    epoch,
                "model":    style_enc.state_dict(),
                "loss":     best_loss,
                "n_tokens": args.n_tokens,
            }, str(out_dir / "style_encoder_best.pt"))

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model": style_enc.state_dict(),
            }, str(out_dir / f"style_encoder_epoch{epoch:04d}.pt"))

    logger.info(f"Done. Best loss={best_loss:.4f}")
    logger.info(f"Saved: {out_dir}/style_encoder_best.pt")


if __name__ == "__main__":
    main()
