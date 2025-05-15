import os
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from models.simclr_segmenter import SimCLREncoder
from data_pipeline.generate_all_datasets import generate_segment_dataset

# === Configuration ===
CONFIG = {
    "dataset_path": "demo_data/segment_dataset.json",
    "checkpoint_dir": "checkpoints",
    "checkpoint_file": "simclr_segmenter_model.pt",
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-3,
    "projection_dim": 32,
    "input_dim": 16,  # Will be overwritten by actual data shape
    "temperature": 0.5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "early_stopping_patience": 10
}

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Set Seeds ===
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# === Contrastive Loss Function (NT-Xent) ===
def nt_xent_loss(zi, zj, temperature):
    batch_size = zi.shape[0]
    z = torch.cat([zi, zj], dim=0)
    z = nn.functional.normalize(z, dim=1)

    similarity_matrix = torch.matmul(z, z.T)
    labels = torch.arange(batch_size).to(zi.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(batch_size * 2, dtype=torch.bool).to(zi.device)
    similarity_matrix = similarity_matrix[~mask].view(batch_size * 2, -1)

    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)])
    negatives = similarity_matrix

    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(batch_size * 2, dtype=torch.long).to(zi.device)

    logits = logits / temperature
    loss = nn.CrossEntropyLoss()(logits, labels)

    return loss

# === Dataset for SimCLR Segmenter ===
class SegmentDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.samples = [torch.tensor(sample["segment_sequence"], dtype=torch.float32) for sample in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # SimCLR requires two augmentations per sample
        sample = self.samples[idx]
        aug1 = sample + 0.01 * torch.randn_like(sample)  # Simple Gaussian noise augmentation
        aug2 = sample + 0.01 * torch.randn_like(sample)
        return aug1, aug2

# === Training Loop ===
def train_simclr_segmenter():
    logger.info("🚀 Starting SimCLR Segmenter Training Pipeline...")

    # Load or regenerate dataset
    if not os.path.exists(CONFIG["dataset_path"]):
        logger.info(f"⚠️ Dataset not found at {CONFIG['dataset_path']}, regenerating...")
        segment_data = generate_segment_dataset()
        with open(CONFIG["dataset_path"], "w") as f:
            json.dump(segment_data, f, indent=2)
    else:
        logger.info(f"✅ Loading dataset from {CONFIG['dataset_path']}")
        with open(CONFIG["dataset_path"], "r") as f:
            segment_data = json.load(f)

    logger.info(f"✅ Loaded {len(segment_data)} samples for training.")

    # Prepare DataLoader
    dataset = SegmentDataset(segment_data)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    CONFIG["input_dim"] = dataset[0][0].shape[0]
    logger.info(f"✅ Input dimension set to {CONFIG['input_dim']}, Projection dim: {CONFIG['projection_dim']}")

    # Model & Optimizer
    model = SimCLREncoder(input_dim=CONFIG["input_dim"], projection_dim=CONFIG["projection_dim"]).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # Training Vars
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []

    # Training Loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0

        for aug1, aug2 in dataloader:
            aug1, aug2 = aug1.to(CONFIG["device"]), aug2.to(CONFIG["device"])
            zi = model(aug1)
            zj = model(aug2)

            loss = nt_xent_loss(zi, zj, CONFIG["temperature"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        logger.info(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {avg_loss:.6f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"]))
            logger.info(f"💾 Saved best model at {CONFIG['checkpoint_file']}")
        else:
            patience_counter += 1
            logger.info(f"⏳ EarlyStopping Counter: {patience_counter}/{CONFIG['early_stopping_patience']}")
            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info(f"⛔ Early stopping triggered after {epoch+1} epochs.")
                break

    # Final Model Save
    final_model_path = os.path.join(CONFIG["checkpoint_dir"], "simclr_segmenter_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Final model saved at {final_model_path}")

    # Plot Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Contrastive Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("SimCLR Segmenter Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "simclr_training_plot.png"))
    logger.info(f"📈 Training loss plot saved as simclr_training_plot.png")


if __name__ == "__main__":
    train_simclr_segmenter()
