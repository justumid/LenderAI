import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyEstimator(nn.Module):
    """
    Lightweight confidence scoring model.
    Learns to output trust scores for downstream risk predictions.
    Can also perform MC Dropout-based uncertainty estimation.
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 32, dropout: float = 0.2):
        super(UncertaintyEstimator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in range [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to produce confidence score.
        Args:
            x: [B, input_dim] feature vector
        Returns:
            [B] confidence in [0.0 – 1.0]
        """
        return self.net(x).squeeze(-1)

    def mc_predict(self, x: torch.Tensor, n_samples: int = 10) -> (torch.Tensor, torch.Tensor):
        """
        Performs MC Dropout sampling to estimate uncertainty.
        Run multiple forward passes in training mode to simulate stochastic dropout.

        Args:
            x: [B, D] input
            n_samples: how many times to sample

        Returns:
            Tuple:
              mean: [B] average prediction
              std:  [B] standard deviation (uncertainty)
        """
        self.train()  # Enable dropout during inference
        preds = []

        for _ in range(n_samples):
            preds.append(self.forward(x))

        self.eval()  # Return to eval mode

        stacked = torch.stack(preds)  # [T, B]
        return stacked.mean(dim=0), stacked.std(dim=0)
