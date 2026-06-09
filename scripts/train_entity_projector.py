"""
Phase 2: Train EntityProjector with triplet + contrastive loss.

Usage:
  python scripts/train_entity_projector.py \
    --data-dir data/pema_train \
    --out-dir outputs/runs/pema_phase2
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.memory.entity_encoder import EntityEncoder
from src.model.entity_projector import EntityProjector, triplet_loss, contrastive_loss
from src.data.pema_dataset import EntityPairDataset
from src.utils.logging import get_logger

logger = get_logger("train_projector")


def collate(batch):
    return {
        "anchor":   torch.stack([b["anchor"] for b in batch]),
        "positive": torch.stack([b["positive"] for b in batch]),
        "negative": torch.stack([b["negative"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/pema_train")
    parser.add_argument("--out-dir", default="outputs/runs/pema_phase2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lambda-contrastive", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    out_dir = base / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Entity encoder (CLIP ViT-H/14, same as IP-Adapter)
    logger.info("Loading EntityEncoder (CLIP ViT-H/14)...")
    encoder = EntityEncoder(device=str(device))

    # Dataset
    data_dir = base / args.data_dir
    entity_list_path = data_dir / "entity_list.txt"
    entity_list = entity_list_path.read_text().splitlines()
    logger.info(f"Entities: {len(entity_list)}")

    dataset = EntityPairDataset(
        img_dir=str(data_dir / "images"),
        entity_list=entity_list,
        encoder=encoder,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate)

    # Model
    projector = EntityProjector(dim=1024, hidden=2048).to(device)
    n_params = sum(p.numel() for p in projector.parameters())
    logger.info(f"EntityProjector params: {n_params/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        projector.train()
        epoch_loss = 0.0
        for batch in loader:
            anchor   = batch["anchor"].to(device)
            positive = batch["positive"].to(device)
            negative = batch["negative"].to(device)

            proj_a = projector(anchor)
            proj_p = projector(positive)
            proj_n = projector(negative)

            # Triplet: anchor closer to positive than negative
            l_triplet = triplet_loss(proj_a, proj_p, proj_n, margin=args.margin)

            # Contrastive on (anchor, positive) same-entity pairs
            labels = torch.ones(anchor.size(0), device=device)
            l_cont = contrastive_loss(proj_a, proj_p, labels, margin=0.1)

            loss = l_triplet + args.lambda_contrastive * l_cont

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save({
                "epoch": epoch,
                "model": projector.state_dict(),
                "loss": best_loss,
                "dim": 1024,
            }, str(out_dir / "projector_best.pt"))

        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch,
                "model": projector.state_dict(),
                "dim": 1024,
            }, str(out_dir / f"projector_epoch{epoch:03d}.pt"))

    logger.info(f"Training done. Best loss={best_loss:.4f}")
    logger.info(f"Saved: {out_dir}/projector_best.pt")


if __name__ == "__main__":
    main()
