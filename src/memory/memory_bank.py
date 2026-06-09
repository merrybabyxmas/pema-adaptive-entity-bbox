"""Entity Memory Bank: canonical + recent memory per entity with quality-gated EMA update."""
from __future__ import annotations
import torch
from PIL import Image

from .entity_encoder import EntityEncoder


class EntityMemoryBank:
    """
    M_e = {M_e^can, M_e^recent}

    canonical: stable identity prototype from ref/bootstrap image (fixed)
    recent:    EMA of recent shot crops, gated by quality score
    """

    def __init__(self, encoder: EntityEncoder,
                 alpha: float = 0.2,
                 quality_threshold: float = 0.5,
                 can_weight: float = 0.8,
                 rec_weight: float = 0.2,
                 drift_floor: float = 0.6):
        self.encoder = encoder
        self.alpha = alpha
        self.quality_threshold = quality_threshold
        # Canonical-dominant fusion: the bootstrap/ref identity is the anchor,
        # recent only mildly adapts. Heavy recent weight + EMA was observed to
        # drift identity across shots (cat whitened by shot 2).
        self.can_weight = can_weight
        self.rec_weight = rec_weight
        # After EMA, if recent strays below this cosine-sim to canonical, pull
        # it back toward canonical — bounds runaway drift while allowing mild
        # pose/appearance adaptation.
        self.drift_floor = drift_floor

        self._canonical: dict[str, torch.Tensor] = {}
        self._recent: dict[str, torch.Tensor] = {}
        self._ref_images: dict[str, Image.Image] = {}

    def initialize(self, entity_name: str, ref_image: Image.Image) -> None:
        tokens = self.encoder.encode(ref_image)
        self._canonical[entity_name] = tokens
        self._recent[entity_name] = tokens.clone()
        self._ref_images[entity_name] = ref_image.copy()

    def has(self, entity_name: str) -> bool:
        return entity_name in self._canonical

    def retrieve_tokens(self, entity_name: str) -> torch.Tensor:
        """Fused canonical + recent token vector, L2-normalized.

        Normalization keeps the conditioning on the unit sphere regardless of
        how much `recent` has drifted in magnitude — matching the training
        distribution (entity tokens are L2-normalized in the dataset).
        """
        can = self._canonical[entity_name]
        rec = self._recent.get(entity_name, can)
        fused = self.can_weight * can + self.rec_weight * rec
        return fused / (fused.norm() + 1e-8)

    def retrieve_image(self, entity_name: str) -> Image.Image:
        """Canonical reference image (passed to GLIGEN gligen_images)."""
        return self._ref_images[entity_name]

    def update(self, entity_name: str, crop: Image.Image) -> bool:
        """
        Update recent memory with a shot crop.
        Returns True if update was accepted (quality gate passed).
        """
        if entity_name not in self._canonical:
            return False
        u = self.encoder.encode(crop)
        can = self._canonical[entity_name]
        sim = self.encoder.similarity(u, can)
        if sim < self.quality_threshold:
            return False  # reject: too far from canonical identity
        rec = (1 - self.alpha) * self._recent[entity_name] + self.alpha * u
        # Drift clamp: if the updated recent strays too far from canonical,
        # blend it back toward canonical so identity cannot run away over shots.
        if self.encoder.similarity(rec, can) < self.drift_floor:
            rec = 0.5 * rec + 0.5 * can
        self._recent[entity_name] = rec
        return True

    def entities(self) -> list[str]:
        return list(self._canonical.keys())
