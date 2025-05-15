import torch
import torch.nn as nn
from typing import Dict, Union, Any


class LGDEstimator(nn.Module):
    """
    Loss Given Default (LGD) predictor:
    - Predicts percent of loss if borrower defaults.
    - Output always in range [0.05, 1.0].
    - Uses MLP + rule fallback + behavior heuristics.
    """

    def __init__(self, input_dim: int = 12):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicts LGD in range [0.05, 1.0]
        """
        raw = self.model(x).squeeze(-1)
        return 0.05 + raw * 0.95  # scale into valid range

    def predict(self, features: Dict[str, Union[float, int]]) -> float:
        """
        Rule-based fallback LGD estimate from raw features.
        Includes behavioral and structural signals.
        """

        salary = features.get("salary_mean", 0.0)
        open_debt = features.get("total_open_debt", 0.0)
        overdue_ratio = features.get("open_overdue_ratio", 0.0)
        consistency = features.get("repayment_consistency_ratio", 0.0)
        volatility = features.get("repayment_volatility", 1.0)
        contracts = features.get("num_open_contracts", 0)
        katm_score = features.get("katm_score", 0)
        salary_growth = features.get("salary_growth", 0.0)
        ptir = features.get("avg_payment_to_income_ratio", 1.0)
        dti = features.get("dti_ratio", 1.0)
        overdue_qty = features.get("overview_overdue_principal_qty", 0)
        ead = features.get("ead", open_debt)

        explanation = {}

        # === Start with base LGD from overdue status
        base = 0.6
        if overdue_ratio > 0.5:
            base += 0.2
        elif overdue_ratio > 0.2:
            base += 0.1
        elif overdue_ratio < 0.05:
            base -= 0.1

        # === Behavioral Modifiers
        if consistency > 0.95:
            base *= 0.8
        elif consistency < 0.6:
            base *= 1.3

        if volatility > 0.4:
            base *= 1.15
        if salary_growth < 0:
            base *= 1.1

        # === Exposure-Based Adjustments
        if open_debt > 100_000_000:
            base *= 1.15
        elif open_debt < 30_000_000:
            base *= 0.9

        # === Diversification
        if contracts >= 5:
            base *= 0.9
        elif contracts <= 1:
            base *= 1.1

        # === KATM Score Modifier
        if katm_score >= 450:
            base *= 0.8
        elif katm_score <= 350:
            base *= 1.2

        # === DTI/PTI sanity
        if dti > 0.6 or ptir > 0.6:
            base *= 1.2
        elif dti < 0.3 and ptir < 0.3:
            base *= 0.85

        # === Overdue qty
        if overdue_qty >= 3:
            base += 0.05

        # === EAD fallback
        if ead and open_debt > ead:
            base *= 1.05

        # === Clamp and return
        final_lgd = min(max(base, 0.05), 1.0)

        return round(final_lgd, 4)
