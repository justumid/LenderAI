import shap
import joblib
import numpy as np
from typing import Dict, Any, List, Optional

class ExplanationModel:
    """
    Advanced Explanation Module:
    - SHAP-based explanations for PD, Fraud, Limit models
    - Rule-based summary fallback
    - Feature attribution for visualizations
    """

    def __init__(self,
                 pd_model_path="./checkpoints/pd_model.pkl",
                 limit_model_path="./checkpoints/limit_model.pkl",
                 fraud_model_path="./checkpoints/fraud_model.pkl",
                 scaler_path="./checkpoints/explain_scaler.pkl"):

        # Load trained models
        self.pd_model = joblib.load(pd_model_path)
        self.limit_model = joblib.load(limit_model_path)
        self.fraud_model = joblib.load(fraud_model_path)

        # Load Scaler
        self.scaler = joblib.load(scaler_path)

        # SHAP Explainers
        self.pd_explainer = shap.Explainer(self.pd_model)
        self.limit_explainer = shap.Explainer(self.limit_model)
        self.fraud_explainer = shap.Explainer(self.fraud_model)

        # Thresholds for summary rules
        self.thresholds = {
            "pd": {"high": 0.7, "medium": 0.4},
            "fraud": 0.4,
            "static": {"low": 0.4, "high": 0.75}
        }

    def explain_features(self, model_type: str, features: Dict[str, float]) -> List[str]:
        input_vector = [features.get(k, 0.0) for k in self.scaler.feature_names_in_]
        scaled_vector = self.scaler.transform([input_vector])

        if model_type == "pd":
            shap_values = self.pd_explainer(scaled_vector)
        elif model_type == "limit":
            shap_values = self.limit_explainer(scaled_vector)
        elif model_type == "fraud":
            shap_values = self.fraud_explainer(scaled_vector)
        else:
            raise ValueError("Unsupported model_type for explain_features")

        explanation = []
        feature_names = self.scaler.feature_names_in_

        shap_vals = shap_values.values[0]
        sorted_items = sorted(zip(feature_names, shap_vals), key=lambda kv: abs(kv[1]), reverse=True)

        for feature, value in sorted_items[:5]:
            if value > 0:
                explanation.append(f"{feature} increased risk (+{value:.3f})")
            else:
                explanation.append(f"{feature} decreased risk ({value:.3f})")

        return explanation

    def explain_segment(self, segment: str) -> str:
        segment_map = {
            "high_income_reliable": "Strong salary and payment history.",
            "low_income": "Low income flagged affordability risk.",
            "overdue_unstable": "Behavioral instability due to overdue patterns.",
            "bureau_risk": "Low bureau score suggests credit caution.",
            "moderate_risk": "Standard profile with balanced risk."
        }
        return segment_map.get(segment, f"Segment: {segment} — standard profile.")

    def summary_reason(self, pd: float, fraud: float, static_score: float, segment: str) -> str:
        if fraud >= self.thresholds["fraud"]:
            return "High fraud anomaly detected — refer to manual review."
        if pd >= self.thresholds["pd"]["high"]:
            return "Default probability is high — rejection likely."
        if static_score <= self.thresholds["static"]["low"]:
            return "Low static score — insufficient for approval."
        if segment in ["overdue_unstable", "bureau_risk"]:
            return "Behavioral signals suggest review required."
        return "No critical risks detected — approvable."

    def explain_limit_decision(self, limit: float, requested: float) -> str:
        if limit == 0.0:
            return "Loan limit denied due to high risk or insufficient capacity."
        elif limit < requested * 0.5:
            return f"Loan approved but below requested ({limit:,.0f} UZS)."
        else:
            return f"Loan approved for {limit:,.0f} UZS."

    def global_feature_importance(self, model_type: str, dataset_sample: np.ndarray) -> shap.Explanation:
        """
        For global dashboard visualizations.
        """
        if model_type == "pd":
            return self.pd_explainer(dataset_sample)
        elif model_type == "limit":
            return self.limit_explainer(dataset_sample)
        elif model_type == "fraud":
            return self.fraud_explainer(dataset_sample)
        else:
            raise ValueError("Unsupported model_type for global_feature_importance")
