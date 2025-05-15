# models/loss_functions.py

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for:
      - PD (Binary Classification)
      - EAD (Regression)
      - LGD (Regression)
      - Fraud Score (Binary Classification)
      - Loan Limit (Regression)
    """

    def __init__(self,
                 pd_weight: float = 1.0,
                 ead_weight: float = 0.5,
                 lgd_weight: float = 0.5,
                 fraud_weight: float = 1.0,
                 limit_weight: float = 0.5):
        super().__init__()

        self.pd_loss = nn.BCELoss()
        self.ead_loss = nn.MSELoss()
        self.lgd_loss = nn.MSELoss()
        self.fraud_loss = nn.BCELoss()
        self.limit_loss = nn.MSELoss()

        self.weights = {
            "pd": pd_weight,
            "ead": ead_weight,
            "lgd": lgd_weight,
            "fraud": fraud_weight,
            "limit": limit_weight
        }

    def forward(
        self,
        preds: dict,
        targets: dict
    ) -> torch.Tensor:
        """
        preds: {
          'pd', 'ead', 'lgd', 'fraud', 'loan_limit'
        }
        targets: {
          'pd', 'ead', 'lgd', 'fraud', 'loan_limit'
        }
        """
        loss = 0.0

        # PD binary classification
        loss_pd = self.pd_loss(preds["pd"], targets["pd"])
        loss += self.weights["pd"] * loss_pd

        # EAD regression
        loss_ead = self.ead_loss(preds["ead"], targets["ead"])
        loss += self.weights["ead"] * loss_ead

        # LGD regression
        loss_lgd = self.lgd_loss(preds["lgd"], targets["lgd"])
        loss += self.weights["lgd"] * loss_lgd

        # Fraud binary classification
        loss_fraud = self.fraud_loss(preds["fraud"], targets["fraud"])
        loss += self.weights["fraud"] * loss_fraud

        # Loan Limit regression
        loss_limit = self.limit_loss(preds["loan_limit"], targets["loan_limit"])
        loss += self.weights["limit"] * loss_limit

        return loss

if __name__ == "__main__":
    # Example smoke test
    loss_fn = MultiTaskLoss()

    preds = {
        "pd": torch.sigmoid(torch.randn(8)),
        "ead": torch.rand(8),
        "lgd": torch.rand(8),
        "fraud": torch.sigmoid(torch.randn(8)),
        "loan_limit": torch.rand(8) * 100_000_000
    }

    targets = {
        "pd": torch.randint(0, 2, (8,), dtype=torch.float32),
        "ead": torch.rand(8),
        "lgd": torch.rand(8),
        "fraud": torch.randint(0, 2, (8,), dtype=torch.float32),
        "loan_limit": torch.rand(8) * 100_000_000
    }

    loss_value = loss_fn(preds, targets)
    print(f"Multi-task Loss: {loss_value.item():.4f}")
