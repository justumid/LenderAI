import logging
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SimCLREncoder(nn.Module):
    def __init__(self, input_dim=20, embedding_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)

class SegmentClusterer:
    """
    Hybrid Segment Clusterer:
    - SimCLR embeddings from behavior sequences
    - KMeans clustering on embedding space
    - Rule-based fallback logic
    """

    def __init__(self, simclr_model_path="./checkpoints/simclr_encoder.pt",
                       kmeans_path="./checkpoints/segment_kmeans.pkl",
                       scaler_path="./checkpoints/segment_scaler.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load SimCLR Encoder
        self.simclr = SimCLREncoder().to(self.device)
        self.simclr.load_state_dict(torch.load(simclr_model_path, map_location=self.device))
        self.simclr.eval()

        # Load Scaler & KMeans
        self.scaler = joblib.load(scaler_path)
        self.kmeans = joblib.load(kmeans_path)

    def predict(self, features: Dict[str, float], sequences: Dict[str, list]) -> str:
        try:
            embedding_input = self._prepare_embedding_input(features, sequences)
            x = torch.tensor(embedding_input, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.simclr(x).cpu().numpy()

            label = self.kmeans.predict(embedding)[0]
            return f"cluster_{label}"

        except Exception as e:
            logger.warning(f"SegmentClusterer fallback due to: {e}")
            return self._fallback_segment(features)

    def _prepare_embedding_input(self, features: Dict[str, float], sequences: Dict[str, list]) -> np.ndarray:
        static_vector = [features.get(k, 0.0) for k in self.feature_keys()]
        debt_sequence = sequences.get("debt_delta", [])[:10]
        repayment_sequence = sequences.get("repayment_delta", [])[:10]
        combined = static_vector + debt_sequence + repayment_sequence
        combined = combined + [0.0] * (20 - len(combined))  # pad to 20 if short
        return np.array(combined, dtype=np.float32)

    def feature_keys(self) -> list:
        return [
            "salary_mean",
            "repayment_consistency_ratio",
            "open_overdue_ratio",
            "katm_score",
            "avg_payment_to_income_ratio",
            "credit_burden_index",
            "num_credit_requests",
            "salary_growth",
            "salary_std",
            "fraud_score"
        ]

    def _fallback_segment(self, features: Dict[str, float]) -> str:
        salary = features.get("salary_mean", 0)
        consistency = features.get("repayment_consistency_ratio", 0)
        overdue_ratio = features.get("open_overdue_ratio", 0)
        katm_score = features.get("katm_score", 0)

        if salary > 7_000_000 and consistency > 0.95:
            return "high_income_reliable"
        elif salary < 1_500_000:
            return "low_income"
        elif overdue_ratio > 0.3:
            return "overdue_unstable"
        elif katm_score < 100:
            return "bureau_risk"
        return "moderate_risk"
