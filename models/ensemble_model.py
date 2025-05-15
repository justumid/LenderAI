import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class EnsembleModel(nn.Module):
    """
    Combines multiple model outputs (PD, LGD, Fraud, Static, etc.)
    into a unified risk score and classification.
    Can use either:
    - Weighted sum (transparent, interpretable)
    - MLP head (trainable stacker)
    """
import torch
import torch.nn as nn
from typing import Dict, Any

class EnsembleModel(nn.Module):
    """
    Ensemble model that learns to combine PD, LGD, static score, and fraud score dynamically.
    """

    def __init__(self, input_dim: int = 4, hidden_dim: int = 16):
        super(EnsembleModel, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Optional: You can also initialize explicit learnable weights for weighted sum ensemble
        self.learnable_weights = nn.Parameter(torch.ones(input_dim) / input_dim)

    def forward(self, pd: torch.Tensor, lgd: torch.Tensor, static_score: torch.Tensor, fraud_score: torch.Tensor) -> torch.Tensor:
        """
        Combines input scores into a final ensemble score.

        Inputs:
            pd: [batch_size]
            lgd: [batch_size]
            static_score: [batch_size]
            fraud_score: [batch_size]

        Output:
            final_score: [batch_size]
        """

        # Stack into shape [batch_size, 4]
        inputs = torch.stack([pd, lgd, static_score, fraud_score], dim=1)

        # Option 1: Dynamic Weighted Sum (learnable weights softmax normalized)
        weights = torch.softmax(self.learnable_weights, dim=0)  # [4]
        weighted_score = (inputs * weights).sum(dim=1, keepdim=True)  # [batch_size, 1]

        # Option 2: Learn nonlinear interactions via MLP
        mlp_output = self.linear(inputs)  # [batch_size, 1]

        # Final score: Average or weighted combination of both
        final_score = 0.5 * weighted_score + 0.5 * mlp_output  # You can tune this ratio

        return final_score.squeeze(-1)  # [batch_size]

    def classify_risk(self, scores: torch.Tensor) -> list:
        """
        Risk classification from score:
            > 0.7   → High
            > 0.4   → Medium
            ≤ 0.4   → Low
        """
        results = []
        for s in scores:
            val = s.item()
            if val > 0.7:
                results.append("high")
            elif val > 0.4:
                results.append("medium")
            else:
                results.append("low")
        return results
