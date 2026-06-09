"""
Train EntityMemoryConditioner to align CLIP visual embeddings (1024d)
with CLIP text encoder pooler_output (768d) for entity phrases.

Objective: conditioner(CLIP_img(entity)) ≈ text_enc(entity_name).pooler
This allows entity memory to seamlessly replace text grounding tokens in
MemoryGLIGENPipeline.

Data: entity images from data/pema_train/images/ + entity_list.txt
Positive pairs: same entity name → image embedding ↔ text embedding
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.memory.entity_encoder import EntityEncoder
from src.model.entity_memory_conditioner import (
    EntityMemoryConditioner, conditioner_alignment_loss
)
from src.utils.logging import get_logger

logger = get_logger("train_conditioner")


class EntityAlignDataset(Dataset):
    """
    Returns (clip_img_emb, clip_text_emb) pairs for entity images.
    clip_img_emb: CLIP ViT-H/14 visual pooled embedding (1024d)
    clip_text_emb: CLIP text encoder pooler_output for entity name (768d)
    """

    def __init__(self, img_dir: str, entity_list: list[str],
                 img_encoder: EntityEncoder, text_encoder, tokenizer,
                 device: str):
        self.img_encoder = img_encoder
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.device = device

        import re
        def slug(name): return re.sub(r"[^a-z0-9_]", "_", name.lower().strip())

        img_root = Path(img_dir)
        self.items = []  # (entity_name, image_path)
        for entity in entity_list:
            paths = sorted((img_root / slug(entity)).glob("*.png"))
            for p in paths:
                self.items.append((entity, p))

        # Pre-compute text embeddings (fixed)
        self.text_cache: dict[str, torch.Tensor] = {}
        for entity in entity_list:
            tok = tokenizer(entity, padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                text_emb = text_encoder(**tok).pooler_output.squeeze(0).float()
            self.text_cache[entity] = text_emb.cpu()

        logger.info(f"EntityAlignDataset: {len(self.items)} samples, "
                    f"{len(entity_list)} entities")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        entity, img_path = self.items[idx]
        img = Image.open(str(img_path)).convert("RGB")
        img_emb = self.img_encoder.encode(img).cpu()   # (1024,)
        txt_emb = self.text_cache[entity]               # (768,)
        return {"img_emb": img_emb, "txt_emb": txt_emb, "entity": entity}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/pema_train")
    parser.add_argument("--out-dir", default="outputs/runs/pema_conditioner")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base    = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    # Load encoders
    logger.info("Loading EntityEncoder (CLIP ViT-H/14)...")
    img_encoder = EntityEncoder(device=str(device))

    logger.info("Loading text encoder (SD1.5 CLIP ViT-L/14)...")
    from diffusers import StableDiffusionGLIGENPipeline
    pipe = StableDiffusionGLIGENPipeline.from_pretrained(
        "masterful/gligen-1-4-generation-text-box",
        torch_dtype=torch.float16
    ).to(str(device))
    text_encoder = pipe.text_encoder
    tokenizer    = pipe.tokenizer
    text_encoder.eval()
    del pipe  # free GPU memory except text_encoder

    # Dataset
    data_dir    = base / args.data_dir
    entity_list = (data_dir / "entity_list.txt").read_text().splitlines()

    dataset = EntityAlignDataset(
        img_dir=str(data_dir / "images"),
        entity_list=entity_list,
        img_encoder=img_encoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=str(device),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0)

    # Model
    conditioner = EntityMemoryConditioner(in_dim=1024, out_dim=768).to(device)
    n_params = sum(p.numel() for p in conditioner.parameters())
    logger.info(f"EntityMemoryConditioner params: {n_params/1e6:.2f}M")

    optimizer = torch.optim.AdamW(conditioner.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        conditioner.train()
        total_loss = 0.0
        for batch in loader:
            img_emb = batch["img_emb"].to(device)  # (B, 1024)
            txt_emb = batch["txt_emb"].to(device)  # (B, 768)

            pred = conditioner(img_emb)             # (B, 768)
            loss = conditioner_alignment_loss(pred, txt_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conditioner.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | loss={avg:.4f} "
            f"| lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch": epoch,
                "model": conditioner.state_dict(),
                "loss": best_loss,
            }, str(out_dir / "conditioner_best.pt"))

        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": conditioner.state_dict(),
            }, str(out_dir / f"conditioner_epoch{epoch:03d}.pt"))

    logger.info(f"Done. Best loss={best_loss:.4f}")
    logger.info(f"Saved: {out_dir}/conditioner_best.pt")


if __name__ == "__main__":
    main()
