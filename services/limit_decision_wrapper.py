import torch
import logging
from typing import Dict, Any

from models.limit_decision_model import LimitDecisionModel
from services.limit_decision_service import LimitDecisionService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CombinedLimitDecisionEngine:
    """
    Combines ML limit prediction with rule-based LimitDecisionService logic.
    """

    def __init__(self, model_input_dim: int = 12, max_limit: float = 300_000_000):
        self.model = LimitDecisionModel(input_dim=model_input_dim, max_limit=max_limit)
        self.service = LimitDecisionService()
        self.max_limit = max_limit

    def decide_limit(
        self,
        features: Dict[str, Any],
        pd_score: float,
        ead: float,
        lgd: float,
        segment: str,
        static_score: float,
        fraud_score: float,
        use_model: bool = True
    ) -> Dict[str, Any]:
        """
        Combines ML and rule-based limit decision.
        Returns:
            {
                "final_limit": float,
                "model_limit": float,
                "service_limit": float,
                "explanation": dict
            }
        """
        model_limit = None

        # === Step 1: ML Limit Prediction ===
        if use_model:
            try:
                input_vector = self._prepare_input_tensor(features)
                self.model.eval()
                with torch.no_grad():
                    model_output = self.model(input_vector).item()
                    model_limit = model_output * ead  # ML predicts multiplier to EAD
                    logger.info(f"ML Model Limit Prediction: {model_limit:.2f}")
            except Exception as e:
                logger.error(f"[CombinedLimitDecisionEngine] ML prediction failed: {e}")
                model_limit = None

        # === Step 2: Rule-based Service Decision ===
        service_limit, explanation = self.service.decide_segmented(
            features=features,
            pd_score=pd_score,
            ead=ead,
            lgd=lgd,
            segment=segment,
            static_score=static_score,
            fraud_score=fraud_score
        )
        logger.info(f"Service Limit Decision: {service_limit:.2f}")

        # === Step 3: Final Limit Fusion ===
        if model_limit is not None:
            final_limit = 0.7 * service_limit + 0.3 * model_limit
        else:
            final_limit = service_limit

        # === Step 4: Cap Limit & Safety ===
        final_limit = min(final_limit, self.max_limit)
        requested_amount = features.get("requested_amount", ead or 100_000_000)
        final_limit = min(final_limit, requested_amount * 1.2)
        final_limit = round(final_limit, -4)

        logger.info(f"Final Combined Limit: {final_limit:.2f}")

        return {
            "final_limit": final_limit,
            "model_limit": round(model_limit or 0.0, -4),
            "service_limit": round(service_limit or 0.0, -4),
            "explanation": explanation
        }

    def _prepare_input_tensor(self, features: Dict[str, Any]) -> torch.Tensor:
        input_keys = [
            "salary_mean", "avg_payment_to_income_ratio", "katm_score_normalized",
            "repayment_consistency_ratio", "repayment_volatility", "num_early_repayments",
            "active_loans_count", "credit_age_days", "dti_ratio",
            "overview_actual_avg_payment", "salary_growth", "overview_overdue_principal_qty"
        ]
        values = [float(features.get(k, 0.0)) for k in input_keys]
        return torch.tensor([values], dtype=torch.float32)
