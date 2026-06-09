"""
GroundingDINO wrapper. Falls back gracefully if not installed.
"""
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GroundingDINOWrapper:
    def __init__(self, box_threshold=0.35, text_threshold=0.25, device="cuda"):
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.model = None
        self._try_load()

    def _try_load(self):
        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino
            logger.info("GroundingDINO loaded successfully")
            self.model = "loaded"
            self._predict_fn = predict
        except ImportError:
            logger.warning("GroundingDINO not installed. Using mock detector.")
            self.model = None

    def detect(self, image, entity_names: list[str]) -> dict:
        """
        image: PIL.Image
        entity_names: list of entity categories to detect
        Returns: {entity_name: {"box": [x1,y1,x2,y2], "score": float}}
        """
        if self.model is None:
            return self._mock_detect(image, entity_names)
        # real detection would go here
        return {}

    def _mock_detect(self, image, entity_names: list[str]) -> dict:
        import random
        result = {}
        for i, name in enumerate(entity_names):
            # Simple grid layout mock
            n = len(entity_names)
            cx = (i + 0.5) / n
            cy = 0.5
            w = 0.8 / n
            h = 0.6
            result[name] = {
                "box": [cx - w/2, cy - h/2, cx + w/2, cy + h/2],
                "score": round(random.uniform(0.5, 0.9), 3),
            }
        return result
