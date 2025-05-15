import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from models.financial_bert import FinancialBERT
from models.vae import FraudVAE
from models.static_rule_model import StaticRuleEngine
from models.limit_decision_model import LimitDecisionModel
from models.uncertainty_estimator import UncertaintyEstimator
from models.fairness_adjuster import FairnessAdjuster


class FullModel(nn.Module):
    """
    Full ensemble scoring model integrating:
    - BERT sequence model for PD, LGD, EAD, Limit.
    - VAE fraud detection.
    - Static rule-based scoring.
    - Optional Limit override, Fairness, Uncertainty modules.
    """

    def __init__(
        self,
        sequence_input_dim: int,
        sequence_length: int,
        static_rule_engine: StaticRuleEngine,
        limit_model: Optional[LimitDecisionModel] = None,
        fairness_adjuster: Optional[FairnessAdjuster] = None,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        fraud_threshold: float = 0.15,
        use_static_score: bool = True,
        weights: Optional[Dict[str, float]] = None
    ):
        super().__init__()

        self.financial_model = FinancialBERT(input_dim=sequence_input_dim)
        self.fraud_model = FraudVAE(input_dim=sequence_input_dim, seq_len=sequence_length)
        self.static_engine = static_rule_engine
        self.limit_model = limit_model
        self.fairness_adjuster = fairness_adjuster
        self.uncertainty_estimator = uncertainty_estimator
        self.use_static_score = use_static_score
        self.fraud_threshold = fraud_threshold

        self.weights = weights or {
            "pd": 0.4,
            "static_score": 0.2,
            "lgd": 0.2,
            "fraud_score": 0.2
        }

        self.final_head = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, sequence_tensor: torch.Tensor, static_features: Dict[str, Any]) -> Dict[str, Any]:
        B = sequence_tensor.size(0)
        device = sequence_tensor.device

        # 1. BERT Model Outputs
        bert_out = self.financial_model(sequence_tensor)
        pd, lgd, ead = bert_out["pd"], bert_out["lgd"], bert_out["ead"]
        embedding, bert_limit = bert_out["embedding"], bert_out["loan_limit"]
        final_limit = bert_limit.clone()

        # 2. Static Rule Score
        static_scores = torch.zeros(B, device=device)
        if self.use_static_score:
            for i in range(B):
                static_scores[i] = self.static_engine.score({k: (v[i].item() if isinstance(v, torch.Tensor) else v) for k, v in static_features.items()})

        # 3. Fraud Detection (VAE anomaly score)
        with torch.no_grad():
            fraud_scores = self.fraud_model.anomaly_score(sequence_tensor)
            fraud_flags = fraud_scores > self.fraud_threshold

        # 4. Limit Override (optional)
        if self.limit_model:
            for i in range(B):
                try:
                    fallback_limit = self.limit_model.predict({k: (v[i].item() if isinstance(v, torch.Tensor) else v) for k, v in static_features.items()})
                    final_limit[i] = fallback_limit
                except:
                    pass  # use original limit

        # 5. Combined Scores
        combined_input = torch.stack([pd, lgd, ead, static_scores, fraud_scores], dim=1)
        final_score = self.final_head(combined_input).squeeze(-1)

        weighted_score = (
            self.weights["pd"] * pd +
            self.weights["lgd"] * lgd +
            self.weights["static_score"] * static_scores +
            self.weights["fraud_score"] * fraud_scores
        ).clamp(0.0, 1.0)

        # 6. Fairness Adjustment (optional)
        if self.fairness_adjuster:
            final_score = self.fairness_adjuster.adjust(final_score, static_features)

        # 7. Uncertainty Estimation (optional)
        confidence = torch.ones_like(final_score)
        if self.uncertainty_estimator:
            confidence, _ = self.uncertainty_estimator.mc_predict(combined_input)

        # 8. Risk Classification
        risk_class = ["high" if s > 0.7 else "medium" if s > 0.4 else "low" for s in final_score]

        return {
            "pd": pd,
            "lgd": lgd,
            "ead": ead,
            "fraud_score": fraud_scores,
            "fraud_flag": fraud_flags,
            "static_score": static_scores,
            "bert_limit": bert_limit,
            "final_limit": final_limit,
            "final_score": final_score,
            "weighted_score": weighted_score,
            "risk_class": risk_class,
            "confidence": confidence,
            "embedding": embedding
        }
