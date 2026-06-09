"""
Train ResidualEntityConditioner.

Objective: r_e = f_θ(CLIP_img(entity)) must be:
  1. Clustered by entity identity (SupCon)
  2. Orthogonal to text_enc(entity_name)  (not just text duplication)
  3. Unit-norm (no collapse / explosion)

Loss: L = L_supcon + λ_orth * L_orth + λ_norm * L_norm

Data: data/pema_train/images/{entity_slug}/*.png (5 images per entity × 60 entities)
Positive pair: two different images of the same entity (sampled at each step)
All images are pre-encoded once in __init__ to avoid repeated CLIP calls.

Output: outputs/runs/pema_conditioner_residual/conditioner_best.pt
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image

from src.memory.entity_encoder import EntityEncoder
from src.model.residual_entity_conditioner import (
    ResidualEntityConditioner, residual_conditioner_loss,
)
from src.utils.logging import get_logger

logger = get_logger("train_residual")


# ── Dataset ──────────────────────────────────────────────────────────────────

class EntityPairDataset(Dataset):
    """
    Returns (img_emb_a, img_emb_b, text_emb, entity_id) contrastive pairs.

    img_emb_a/b: CLIP ViT-H/14 embeddings (1024d) of two different images
                 from the same entity.  Sampled randomly each call.
    text_emb:    text_encoder pooler_output (768d) for entity name.
    entity_id:   integer index used as contrastive label.
    """

    def __init__(self, img_dir: str, entity_list: list,
                 img_encoder: EntityEncoder, text_encoder, tokenizer,
                 device: str):
        import re
        def slug(name): return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())

        img_root = Path(img_dir)
        # {entity_id: [emb_tensor, ...]}
        self.groups: dict[int, list[torch.Tensor]] = {}
        self.entity_id_map: dict[str, int] = {}
        self.text_cache: dict[int, torch.Tensor] = {}

        # Pre-encode all images
        logger.info("Pre-encoding entity images (cached for training)...")
        for eid, entity in enumerate(entity_list):
            paths = sorted((img_root / slug(entity)).glob("*.png"))
            if not paths:
                continue
            embs = []
            for p in paths:
                img = Image.open(str(p)).convert("RGB")
                emb = img_encoder.encode(img).cpu()   # (1024,)
                embs.append(emb)
            self.groups[eid] = embs
            self.entity_id_map[entity] = eid

            # Text embedding
            tok = tokenizer(entity, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                txt = text_encoder(**tok).pooler_output.squeeze(0).float().cpu()
            self.text_cache[eid] = txt

        # Flat index: list of entity_ids for __len__ / __getitem__
        # Each entity appears max(1, n_images) times per "epoch"
        self.index = []
        for eid, embs in self.groups.items():
            self.index.extend([eid] * max(1, len(embs)))

        logger.info(
            f"EntityPairDataset: {len(self.index)} samples, "
            f"{len(self.groups)} entities"
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        eid = self.index[idx]
        embs = self.groups[eid]

        if len(embs) >= 2:
            a_idx, b_idx = random.sample(range(len(embs)), 2)
            emb_a, emb_b = embs[a_idx], embs[b_idx]
        else:
            emb_a = emb_b = embs[0]  # only 1 image — trivial positive

        return {
            "img_emb_a": emb_a,                   # (1024,)
            "img_emb_b": emb_b,                   # (1024,)
            "text_emb":  self.text_cache[eid],    # (768,)
            "entity_id": torch.tensor(eid, dtype=torch.long),
        }


# ── Balanced sampler ─────────────────────────────────────────────────────────

class BalancedEntitySampler(Sampler):
    """
    Each batch contains exactly `samples_per_entity` images for each entity.
    Total batch size = n_entities_per_batch × samples_per_entity.

    This guarantees every batch has positive pairs — critical for SupCon.
    """

    def __init__(self, dataset: EntityPairDataset,
                 samples_per_entity: int = 2,
                 n_entities_per_batch: int = 16,
                 n_batches_per_epoch: int = 50):
        self.groups = dataset.groups                         # {eid: [emb, ...]}
        self.spe    = samples_per_entity
        self.epb    = n_entities_per_batch
        self.n_batches = n_batches_per_epoch
        self.entity_ids = list(self.groups.keys())

    def __len__(self):
        return self.n_batches * self.epb * self.spe

    def __iter__(self):
        for _ in range(self.n_batches):
            batch_eids = random.sample(self.entity_ids,
                                       min(self.epb, len(self.entity_ids)))
            for eid in batch_eids:
                n_avail = len(self.groups[eid])
                for _ in range(self.spe):
                    # yield (eid, img_idx) encoded as dataset index
                    # We map back via dataset.index: find positions with this eid
                    yield (eid, random.randint(0, n_avail - 1))


class BalancedEntityDataset(Dataset):
    """
    Dataset that accepts (entity_id, img_idx) tuples from BalancedEntitySampler.
    Returns the same fields as EntityPairDataset but pair_b is always a
    different image from the same entity.
    """

    def __init__(self, base_dataset: "EntityPairDataset"):
        self.groups     = base_dataset.groups
        self.text_cache = base_dataset.text_cache

    def __len__(self):
        return sum(len(v) for v in self.groups.values())

    def __getitem__(self, item):
        # item = (entity_id, img_idx)  produced by BalancedEntitySampler
        eid, idx_a = item
        embs   = self.groups[eid]
        emb_a  = embs[idx_a]

        # Second view: different index if possible
        if len(embs) >= 2:
            other = [i for i in range(len(embs)) if i != idx_a]
            idx_b = random.choice(other)
        else:
            idx_b = idx_a
        emb_b = embs[idx_b]

        return {
            "img_emb_a": emb_a,
            "img_emb_b": emb_b,
            "text_emb":  self.text_cache[eid],
            "entity_id": torch.tensor(eid, dtype=torch.long),
        }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/pema_train")
    parser.add_argument("--out-dir",    default="outputs/runs/pema_conditioner_residual")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--temperature",       type=float, default=0.07)
    parser.add_argument("--lambda-orth",       type=float, default=0.1)
    parser.add_argument("--samples-per-entity",type=int,   default=2,
                        help="Images per entity per batch (>=2 ensures positive pairs)")
    parser.add_argument("--entities-per-batch",type=int,   default=20,
                        help="Distinct entities per batch")
    parser.add_argument("--batches-per-epoch", type=int,   default=50)
    parser.add_argument("--device",            default="cuda")
    args = parser.parse_args()

    base    = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    # ── Load encoders ─────────────────────────────────────────────────────
    logger.info("Loading EntityEncoder (CLIP ViT-H/14)...")
    img_encoder = EntityEncoder(device=str(device))

    logger.info("Loading text encoder (SD1.5 CLIP ViT-L/14)...")
    from diffusers import StableDiffusionGLIGENPipeline
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box",
        torch_dtype=torch.float16,
    ).to(str(device))
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer
    text_encoder.eval()
    del pipe

    # ── Dataset ───────────────────────────────────────────────────────────
    data_dir    = base / args.data_dir
    entity_list = (data_dir / "entity_list.txt").read_text().splitlines()

    base_dataset = EntityPairDataset(
        img_dir=str(data_dir / "images"),
        entity_list=entity_list,
        img_encoder=img_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=str(device),
    )
    # Balanced sampler: guarantees ≥2 images per entity per batch → always has positives
    balanced_ds = BalancedEntityDataset(base_dataset)
    sampler = BalancedEntitySampler(
        base_dataset,
        samples_per_entity=args.samples_per_entity,
        n_entities_per_batch=args.entities_per_batch,
        n_batches_per_epoch=args.batches_per_epoch,
    )
    # batch_size is implicit: entities_per_batch × samples_per_entity
    batch_size = args.entities_per_batch * args.samples_per_entity
    loader = DataLoader(balanced_ds, batch_size=batch_size,
                        sampler=sampler, num_workers=0, drop_last=False)
    logger.info(
        f"Balanced loader: {args.entities_per_batch} entities × "
        f"{args.samples_per_entity} samples = {batch_size} per batch, "
        f"{args.batches_per_epoch} batches/epoch"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    conditioner = ResidualEntityConditioner(in_dim=1024, out_dim=768).to(device)
    n_params = sum(p.numel() for p in conditioner.parameters())
    logger.info(f"ResidualEntityConditioner params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        conditioner.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        conditioner.train()
        total = {"total": 0., "contrastive": 0., "orth": 0.}
        n_batches = 0

        for batch in loader:
            img_a    = batch["img_emb_a"].to(device)    # (B, 1024)
            img_b    = batch["img_emb_b"].to(device)    # (B, 1024)
            text_emb = batch["text_emb"].to(device)     # (B, 768)
            eids     = batch["entity_id"].to(device)    # (B,)

            r_a = conditioner(img_a)                    # (B, 768)
            r_b = conditioner(img_b)                    # (B, 768)

            loss, components = residual_conditioner_loss(
                r_a, r_b, eids, text_emb,
                temperature=args.temperature,
                lambda_orth=args.lambda_orth,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total["total"] += loss.item()
            for k, v in components.items():
                total[k] += v
            n_batches += 1

        avg = {k: v / n_batches for k, v in total.items()}
        logger.info(
            f"Epoch {epoch:4d}/{args.epochs} | "
            f"loss={avg['total']:.4f} "
            f"[cont={avg['contrastive']:.4f} "
            f"orth={avg['orth']:.4f}] "
            f"| lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg["total"] < best_loss:
            best_loss = avg["total"]
            torch.save({
                "epoch": epoch,
                "model": conditioner.state_dict(),
                "loss":  best_loss,
                "components": avg,
            }, str(out_dir / "conditioner_best.pt"))

        if epoch % 20 == 0:
            torch.save({
                "epoch": epoch,
                "model": conditioner.state_dict(),
            }, str(out_dir / f"conditioner_epoch{epoch:04d}.pt"))

    logger.info(f"Done. Best total loss={best_loss:.4f}")
    logger.info(f"Saved: {out_dir}/conditioner_best.pt")


if __name__ == "__main__":
    main()
