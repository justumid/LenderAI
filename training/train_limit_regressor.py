import os
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Dict, Any

from models.limit_decision_model import LimitRegressor
from data_pipeline.generate_all_datasets import generate_limit_dataset

# === Configuration ===
CONFIG = {
    "dataset_path": "demo_data/limit_dataset.json",
    "checkpoint_dir": "checkpoints",
    "checkpoint_file": "limit_regressor_model.pt",
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 5,
    "validation_split": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "mixed_precision": True
}

# === Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Seed Reproducibility ===
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# === Dataset Class ===
class LimitDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = torch.tensor(sample["limit_features"], dtype=torch.float32)
        target = torch.tensor(sample["loan_limit"], dtype=torch.float32)
        return features, target

# === Training Function ===
def train_limit_regressor():
    logger.info("🚀 Starting Limit Regressor Training Pipeline...")

    # Load or regenerate dataset
    if not os.path.exists(CONFIG["dataset_path"]):
        logger.info("⚠️ Dataset not found, regenerating...")
        limit_data = generate_limit_dataset()
        with open(CONFIG["dataset_path"], "w") as f:
            json.dump(limit_data, f, indent=2)
    else:
        logger.info("✅ Loading dataset...")
        with open(CONFIG["dataset_path"], "r") as f:
            limit_data = json.load(f)

    logger.info(f"✅ Loaded {len(limit_data)} samples.")

    # Dataset & Split
    full_dataset = LimitDataset(limit_data)
    val_size = int(CONFIG["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    input_dim = full_dataset[0][0].shape[0]
    logger.info(f"✅ Model input_dim: {input_dim}")

    # Model, Optimizer, Scheduler
    model = LimitRegressor(input_dim=input_dim).to(CONFIG["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)
    criterion = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["mixed_precision"])

    best_r2 = -float('inf')
    patience_counter = 0
    history = {"loss": [], "val_r2": []}

    # === Training Loop ===
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0

        for features, targets in train_loader:
            features, targets = features.to(CONFIG["device"]), targets.to(CONFIG["device"])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=CONFIG["mixed_precision"]):
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # === Validation ===
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(CONFIG["device"])
                outputs = model(features).squeeze()

                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.numpy())

        mse = mean_squared_error(val_targets, val_preds)
        r2 = r2_score(val_targets, val_preds)
        scheduler.step(mse)
        history["val_r2"].append(r2)

        logger.info(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {avg_loss:.6f} | Val R2: {r2:.4f} | Val MSE: {mse:.2f}")

        # Early Stopping & Save Best
        if r2 > best_r2:
            best_r2 = r2
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"]))
            logger.info("💾 Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info("⛔ Early stopping triggered.")
                break

    # Final Save
    final_model_path = os.path.join(CONFIG["checkpoint_dir"], "limit_regressor_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Final model saved at {final_model_path}")

    # Final Metrics
    val_mae = mean_absolute_error(val_targets, val_preds)
    logger.info(f"📈 Final Metrics - R2: {best_r2:.4f} | MSE: {mse:.2f} | MAE: {val_mae:.2f}")

    # === Plot Loss & R2 ===
    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_r2"], label="Val R2")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / R2")
    plt.title("Limit Regressor Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "limit_regressor_training_plot.png"))
    logger.info("📉 Training plot saved.")

if __name__ == "__main__":
    train_limit_regressor()
