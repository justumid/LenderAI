import os
import logging
import torch
from typing import Dict, Any

from models.simclr_segmenter import SimCLREncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMCLR_CHECKPOINT_PATH = "checkpoints/simclr_model.pt"

class SegmentService:
    """
    SegmentService assigns customer segments (e.g., trusted, risky)
    using SimCLR behavioral embeddings.
    """

    def __init__(self, input_dim: int = 10):
        logger.info("Initializing SegmentService...")

        self.simclr = SimCLREncoder(input_dim=input_dim)
        self.simclr_loaded = False

        if os.path.exists(SIMCLR_CHECKPOINT_PATH):
            try:
                self.simclr.load_state_dict(torch.load(SIMCLR_CHECKPOINT_PATH, map_location=DEVICE))
                self.simclr.to(DEVICE).eval()
                self.simclr_loaded = True
                logger.info(f"✅ Loaded SimCLR model from {SIMCLR_CHECKPOINT_PATH}")
            except Exception as e:
                logger.error(f"⚠ Failed to load SimCLR checkpoint: {e}. Using fallback mode.")
                self.simclr = None
        else:
            logger.warning(f"⚠ SimCLR checkpoint not found at {SIMCLR_CHECKPOINT_PATH}. Using fallback mode.")
            self.simclr = None

    def identify_segment(self, features: Dict[str, Any], sequences: Dict[str, Any]) -> str:
        """
        Identifies customer segment based on embeddings.
        Returns: 'trusted', 'neutral', 'risky', etc.
        """
        if self.simclr is None:
            logger.info("🔄 Fallback: Defaulting segment to 'neutral' (no SimCLR model loaded).")
            return 'neutral'

        try:
            # Example: embedding based on additional features (flattened)
            feature_vector = self._prepare_input_tensor(features)
            with torch.no_grad():
                embedding = self.simclr(feature_vector.to(DEVICE))

            # Example: L2 norm based heuristic (this can be improved)
            score = torch.norm(embedding, p=2).item()

            if score < 1.0:
                return 'trusted'
            elif score < 2.0:
                return 'neutral'
            else:
                return 'risky'

        except Exception as e:
            logger.error(f"[SegmentService] Failed to compute segment: {e}")
            return 'neutral'  # Safe fallback

    def _prepare_input_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        """
        Converts tabular features dict to input tensor for SimCLR model.
        """
        exclude_keys = ["__source_flags__", "scoring_class", "scoring_level", "scoring_version",
                        "katm_score_class", "katm_score_level"]
        vector = []
        for k, v in features.items():
            if k in exclude_keys:
                continue
            try:
                vector.append(float(v))
            except:
                vector.append(0.0)
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)  # Shape: (1, input_dim)
