import torch
import torch.nn as nn
import joblib
import numpy as np
from typing import Dict, Union

class LimitMLP(nn.Module):
    def __init__(self, input_dim=20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)

class LimitDecisionModel:
    """
    Predicts safe, personalized loan limits:
    - Uses trained LightGBM / MLP models.
    - Has business-rule fallback logic.
    """

    def __init__(self, mlp_path="./checkpoints/limit_mlp.pt",
                       lgbm_path="./checkpoints/limit_lgbm.pkl",
                       scaler_path="./checkpoints/limit_scaler.pkl",
                       max_limit=300_000_000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_limit = max_limit

        # Load models
        self.mlp = LimitMLP(input_dim=20).to(self.device)
        self.mlp.load_state_dict(torch.load(mlp_path, map_location=self.device))
        self.mlp.eval()

        try:
            self.lgbm_model = joblib.load(lgbm_path)
        except:
            self.lgbm_model = None

        self.scaler = joblib.load(scaler_path)

    def predict_from_mlp(self, features: Dict[str, Union[float, int]]) -> float:
        input_vector = [features.get(k, 0.0) for k in self.scaler.feature_names_in_]
        scaled_vector = self.scaler.transform([input_vector])
        x = torch.tensor(scaled_vector, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            pred = self.mlp(x).item()

        pred = max(0.0, min(pred, self.max_limit))
        return round(pred, 2)

    def predict_from_lgbm(self, features: Dict[str, Union[float, int]]) -> Union[float, None]:
        if not self.lgbm_model:
            return None

        input_vector = [features.get(k, 0.0) for k in self.lgbm_model.feature_names_in_]
        pred = self.lgbm_model.predict([input_vector])[0]
        pred = max(0.0, min(pred, self.max_limit))
        return round(pred, 2)

    def predict_from_rules(self, features: Dict[str, Union[float, int]]) -> float:
        # Reuse your existing rule-based fallback logic here (unchanged)
        salary = features.get("salary_mean", 0.0)
        pti = features.get("avg_payment_to_income_ratio", 1.0)
        katm = features.get("katm_score", 0)
        consistency = features.get("repayment_consistency_ratio", 0.0)
        volatility = features.get("repayment_volatility", 1.0)
        early_repayments = features.get("num_early_repayments", 0)
        num_loans = features.get("active_loans_count", 0)
        credit_age = features.get("credit_age_days", 0)

        expected_payment = 800_000
        max_pti = 0.4
        min_salary = expected_payment / max_pti

        if salary < min_salary or katm < 350:
            return 0.0

        base = salary * 3.5

        if pti > 0.6:
            base *= 0.5
        elif pti > 0.4:
            base *= 0.75

        if consistency > 0.95:
            base *= 1.2
        elif consistency < 0.7:
            base *= 0.7

        if volatility > 0.5:
            base *= 0.6
        elif volatility > 0.3:
            base *= 0.8

        if early_repayments >= 3:
            base *= 1.2
        elif early_repayments == 1:
            base *= 1.05

        if credit_age > 1000:
            base *= 1.1

        if num_loans >= 4:
            base *= 0.75

        if katm >= 450:
            base *= 1.15
        elif katm >= 400:
            base *= 1.05

        return min(base, self.max_limit)

    def predict(self, features: Dict[str, Union[float, int]]) -> float:
        # Prefer LightGBM
        pred_lgbm = self.predict_from_lgbm(features)
        if pred_lgbm is not None and pred_lgbm > 0:
            return pred_lgbm

        # Fallback to MLP
        pred_mlp = self.predict_from_mlp(features)
        if pred_mlp > 0:
            return pred_mlp

        # Fallback to rule-based
        return self.predict_from_rules(features)
