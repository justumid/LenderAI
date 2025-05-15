import logging
from typing import Dict, Any, List, Optional
import torch

from models.explanation_model import ExplanationModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ExplanationService:
    """
    Combines static, model, behavioral, fraud, and SHAP insights into
    a human-readable explanation bundle for credit scoring decisions.
    """

    def __init__(self):
        self.model_explainer = ExplanationModel()
        logger.info("✅ ExplanationService initialized.")

    def generate(
        self,
        features: Dict[str, Any],
        pd_score: float,
        ead_score: float,
        lgd_score: float,
        static_score: float,
        fraud_score: float,
        static_labels: List[str],
        behavior_risk: str,
        shap_values: Optional[Dict[str, float]] = None,
        requested_amount: float = 0.0,
        approved_limit: float = 0.0
    ) -> Dict[str, Any]:
        """
        Generate a full structured explanation.
        Returns:
            {
                "pd_explanation": "...",
                "lgd_explanation": "...",
                "fraud_explanation": "...",
                "static_explanation": "...",
                "behavior_explanation": "...",
                "limit_explanation": "...",
                "summary": "...",
                "shap_feature_importance": [...],
            }
        """
        explanations = {
            "pd_explanation": self.model_explainer.explain_pd(pd_score),
            "lgd_explanation": self.model_explainer.explain_lgd(lgd_score),
            "fraud_explanation": self.model_explainer.explain_fraud(fraud_score),
            "static_explanation": self.model_explainer.explain_static_score(static_score),
            "behavior_explanation": self.model_explainer.explain_behavior_segment(behavior_risk),
            "limit_explanation": self.model_explainer.explain_limit(approved_limit, requested_amount),
            "summary": self.model_explainer.generate_summary(pd_score, fraud_score, static_score, behavior_risk),
            "shap_feature_importance": self.model_explainer.shap_summary(shap_values) if shap_values else []
        }

        return explanations

    def shap_feature_importance(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        feature_names: List[str],
        device: str = "cpu"
    ) -> Dict[str, float]:
        """
        Computes SHAP values for a deep model (optional, expensive).
        Returns:
            Dict[feature_name] = shap_value
        """
        import shap

        model.eval()
        input_tensor = input_tensor.to(device)

        explainer = shap.DeepExplainer(model, input_tensor)
        shap_vals = explainer.shap_values(input_tensor)[0]

        output = {}
        for i, name in enumerate(feature_names):
            output[name] = float(shap_vals[0][i])  # batch[0] SHAP

        return output
