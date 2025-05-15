import os
import json
import logging
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any

from models.vae import FraudVAE

# === Configuration ===
CONFIG = {
    "dataset_path": "data/processed_datasets/fraud_dataset.json",
    "checkpoint_dir": "checkpoints",
    "checkpoint_file": "fraud_vae_model.pt",
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "latent_dim": 16,
    "early_stopping_patience": 10,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42
}

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Set Seeds for Reproducibility ===
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# === Custom Dataset Class ===
class FraudVAEDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.samples = [torch.tensor(sample["fraud_sequence"], dtype=torch.float32) for sample in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# === Loss Function ===
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss, recon_loss.item(), kld_loss.item()

# === Training Loop ===
def train_fraud_vae():
    logger.info("🚀 Starting FraudVAE Training Pipeline...")

    # === Load Dataset ===
    if not os.path.exists(CONFIG["dataset_path"]):
        logger.error(f"❌ Dataset not found at {CONFIG['dataset_path']}. Please run dataset_generator first.")
        exit(1)

    logger.info(f"✅ Loading dataset from {CONFIG['dataset_path']}")
    with open(CONFIG["dataset_path"], "r") as f:
        fraud_data = json.load(f)

    # === DataLoader ===
    dataset = FraudVAEDataset(fraud_data)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    input_dim = dataset[0].shape[0]
    logger.info(f"✅ Input dimension: {input_dim}, Latent dimension: {CONFIG['latent_dim']}")

    # === Model & Optimizer ===
    model = FraudVAE(input_dim=input_dim, latent_dim=CONFIG["latent_dim"], seq_len=input_dim).to(CONFIG["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    best_loss = float('inf')
    patience_counter = 0
    train_losses, recon_losses, kld_losses = [], [], []

    # === Training Loop ===
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss, epoch_recon, epoch_kld = 0.0, 0.0, 0.0

        for batch in dataloader:
            batch = batch.to(CONFIG["device"])
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(batch)
            loss, recon, kld = vae_loss(recon_batch, batch, mu, logvar)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon
            epoch_kld += kld

        avg_loss = epoch_loss / len(dataloader)
        avg_recon = epoch_recon / len(dataloader)
        avg_kld = epoch_kld / len(dataloader)

        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)

        logger.info(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} | Total Loss: {avg_loss:.6f} | Recon: {avg_recon:.6f} | KLD: {avg_kld:.6f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"]))
            logger.info(f"💾 Saved best model to {CONFIG['checkpoint_file']}")
        else:
            patience_counter += 1
            logger.info(f"⏳ EarlyStopping Counter: {patience_counter}/{CONFIG['early_stopping_patience']}")
            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info(f"⛔ Early stopping triggered after {epoch+1} epochs.")
                break

    # Final Save
    final_model_path = os.path.join(CONFIG["checkpoint_dir"], "fraud_vae_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Final FraudVAE model saved at {final_model_path}")

    # Plot Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Total Loss")
    plt.plot(recon_losses, label="Reconstruction Loss")
    plt.plot(kld_losses, label="KLD Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("FraudVAE Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "fraud_vae_training_plot.png"))
    logger.info(f"📈 Training loss plot saved at fraud_vae_training_plot.png")

if __name__ == "__main__":
    train_fraud_vae()
