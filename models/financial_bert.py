import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskHead(nn.Module):
    """
    Risk prediction head from [CLS]-style pooled embedding.
    Outputs:
        - pd (default probability)     → sigmoid [0, 1]
        - lgd (loss given default)     → sigmoid [0, 1]
        - ead (exposure at default)    → softplus [0, ∞)
        - loan_limit (recommended cap) → softplus [0, ∞)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(RiskHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # pd, lgd, ead, loan_limit
        )

    def forward(self, embedding: torch.Tensor) -> dict:
        out = self.mlp(embedding)  # [B, 4]
        return {
            "pd": torch.sigmoid(out[:, 0]),
            "lgd": torch.sigmoid(out[:, 1]),
            "ead": F.softplus(out[:, 2]),
            "loan_limit": F.softplus(out[:, 3])
        }


class FinancialBERT(nn.Module):
    """
    Transformer-based encoder for behavioral sequence modeling.
    Input: [B, T, D]
    Output: pd, lgd, ead, loan_limit, and internal embedding
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super(FinancialBERT, self).__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.head = RiskHead(input_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> dict:
        """
        Args:
            x: [B, T, D] input sequences
            attention_mask: [B, T] where 1 = valid, 0 = pad

        Returns:
            {
                pd: [B],
                lgd: [B],
                ead: [B],
                loan_limit: [B],
                embedding: [B, H]
            }
        """
        x = self.input_proj(x)  # [B, T, H]

        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # flip 1/0 → 0/1
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.encoder(x)

        pooled = x[:, 0, :]  # [CLS]-like token
        out = self.head(pooled)
        out["embedding"] = pooled

        return out
