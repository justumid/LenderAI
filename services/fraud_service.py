import os
import torch
import logging
from typing import Dict, Any

from models.vae import FraudVAE
from models.simclr_segmenter import SimCLREncoder

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CURRENT_MODEL_DIR = "current_model/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FraudService:
    def __init__(self, input_dim: int = 10, seq_len: int = 24):
        logger.info(f"✅ Initializing FraudService with input_dim={input_dim}, seq_len={seq_len}...")

        # VAE Model
        self.vae = FraudVAE(input_dim=input_dim, seq_len=seq_len).to(DEVICE)
        vae_ckpt = os.path.join(CURRENT_MODEL_DIR, "vae_model.pt")
        if os.path.exists(vae_ckpt):
            self.vae.load_state_dict(torch.load(vae_ckpt, map_location=DEVICE))
            logger.info("✅ Loaded VAE model for fraud detection.")
        else:
            logger.warning("⚠ VAE checkpoint not found. Fraud scoring will fallback.")
        self.vae.eval()

        # SimCLR Model
        self.simclr = SimCLREncoder(input_dim=input_dim).to(DEVICE)
        simclr_ckpt = os.path.join(CURRENT_MODEL_DIR, "simclr_model.pt")
        if os.path.exists(simclr_ckpt):
            self.simclr.load_state_dict(torch.load(simclr_ckpt, map_location=DEVICE))
            logger.info("✅ Loaded SimCLR model for behavioral embeddings.")
        else:
            logger.warning("⚠ SimCLR checkpoint not found.")
        self.simclr.eval()

    def score(self, features: Dict[str, Any]) -> float:
        """
        Args:
            features: {
                "x_seq": Tensor (seq_len, input_dim),
                "additional_features": Tensor (n_features,)
            }
        Returns:
            fraud_score: float in [0, 1]
        """
        try:
            x_seq = features["x_seq"]
            if not isinstance(x_seq, torch.Tensor):
                x_seq = torch.tensor(x_seq, dtype=torch.float32)

            if x_seq.ndim == 2:
                x_seq = x_seq.unsqueeze(0)

            if x_seq.shape != (1, 24, 10):
                logger.error(f"[FraudService] Invalid x_seq shape: {x_seq.shape}. Expected (1, 24, 10).")
                return 0.5

            x_seq = x_seq.to(DEVICE)

            additional_features = features.get("additional_features")
            if not isinstance(additional_features, torch.Tensor):
                additional_features = torch.tensor(additional_features, dtype=torch.float32)
            if additional_features.ndim == 1:
                additional_features = additional_features.unsqueeze(0)
            additional_features = additional_features.to(DEVICE)

            # === VAE Scoring ===
            with torch.no_grad():
                recon_x, mu, logvar = self.vae(x_seq)
            recon_loss = torch.mean((x_seq - recon_x) ** 2)
            kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            vae_anomaly_score = recon_loss + kl_div
            vae_score = min(1.0, vae_anomaly_score.item() / 0.05)

            # === SimCLR Embedding ===
            with torch.no_grad():
                simclr_embedding = self.simclr(additional_features)
                simclr_norm = torch.norm(simclr_embedding, p=2).item()
            simclr_score = min(1.0, simclr_norm / 5.0)

            # === Final Fraud Score ===
            fraud_score = 0.7 * vae_score + 0.3 * simclr_score
            fraud_score = min(max(fraud_score, 0.0), 1.0)

            logger.info(f"Fraud Score Computed: VAE={vae_score:.4f}, SimCLR={simclr_score:.4f}, Final={fraud_score:.4f}")
            return fraud_score

        except Exception as e:
            logger.error(f"[FraudService] Scoring failed: {e}")
            return 0.5  # fallback

