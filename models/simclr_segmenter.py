import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLREncoder(nn.Module):
    """
    SimCLREncoder:
    A compact MLP-based encoder used for unsupervised representation learning
    in behavioral clustering and contrastive embedding generation.

    - Supports both projected and base embeddings
    - Ideal for KMeans, SimCLR, t-SNE or segment classifiers
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dims: list = [128, 64],
        projection_dim: int = 16,
        dropout: float = 0.1
    ):
        super(SimCLREncoder, self).__init__()

        # === Encoder (MLP)
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        self.backbone = nn.Sequential(*layers)

        # === Projection Head (SimCLR)
        self.projector = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor, return_projected: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] tabular/sequence-derived feature vector
            return_projected: if True, return contrastive normalized projection

        Returns:
            torch.Tensor: [B, hidden_dim] or [B, projection_dim]
        """
        z = self.backbone(x)  # [B, hidden_dim]

        if return_projected:
            return F.normalize(self.projector(z), dim=-1)  # [B, projection_dim]

        return z  # [B, hidden_dim] for downstream use (KMeans, etc.)
