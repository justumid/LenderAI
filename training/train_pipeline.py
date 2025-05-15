import os
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from typing import List, Dict, Any

from models.bert_sequence_model import PDSequenceClassifier
from data_pipeline.generate_all_datasets import generate_pd_dataset

# === Configuration ===
CONFIG = {
    "dataset_path": "demo_data/pd_dataset.json",
    "checkpoint_dir": "checkpoints",
    "checkpoint_file": "pd_bert_model.pt",
    "batch_size": 32,
    "epochs": 30,
    "learning_rate": 2e-5,
    "gradient_clip_norm": 1.0,
    "early_stopping_patience": 5,
    "validation_split": 0.2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "max_sequence_length": 128,
    "mixed_precision": True
}

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Seed for Reproducibility ===
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CONFIG["device"] == "cuda":
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["seed"])

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# === Dataset Class ===
class PDDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sequence = torch.tensor(sample["pd_sequence"][:CONFIG["max_sequence_length"]], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return sequence, label

# === Evaluation Metrics ===
def evaluate_metrics(labels, preds):
    preds_bin = (preds >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(labels, preds),
        "Accuracy": accuracy_score(labels, preds_bin),
        "F1": f1_score(labels, preds_bin),
        "Precision": precision_score(labels, preds_bin),
        "Recall": recall_score(labels, preds_bin)
    }

# === Training Function ===
def train_pd_bert():
    logger.info("🚀 Starting PD BERT Training Pipeline...")

    # Load or regenerate dataset
    if not os.path.exists(CONFIG["dataset_path"]):
        logger.info("⚠️ Dataset not found, regenerating...")
        pd_data = generate_pd_dataset()
        with open(CONFIG["dataset_path"], "w") as f:
            json.dump(pd_data, f, indent=2)
    else:
        logger.info("✅ Loading dataset...")
        with open(CONFIG["dataset_path"], "r") as f:
            pd_data = json.load(f)

    logger.info(f"✅ Loaded {len(pd_data)} samples.")

    # Dataset & Split
    full_dataset = PDDataset(pd_data)
    val_size = int(CONFIG["validation_split"] * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    input_dim = full_dataset[0][0].shape[0]
    logger.info(f"✅ Model input_dim: {input_dim}")

    # Model & Optimizer & Scheduler
    model = PDSequenceClassifier(input_dim=input_dim).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG["mixed_precision"])

    best_auc = 0.0
    patience_counter = 0
    history = {"loss": [], "val_auc": []}

    # === Training Loop ===
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0

        for sequences, labels in train_loader:
            sequences, labels = sequences.to(CONFIG["device"]), labels.to(CONFIG["device"])

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=CONFIG["mixed_precision"]):
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip_norm"])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        history["loss"].append(avg_loss)

        # === Validation ===
        model.eval()
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(CONFIG["device"])
                outputs = model(sequences).squeeze()
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        metrics = evaluate_metrics(all_labels, np.array(all_preds))
        history["val_auc"].append(metrics["AUC"])

        logger.info(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.6f} | Val AUC: {metrics['AUC']:.4f} | Acc: {metrics['Accuracy']:.4f} | F1: {metrics['F1']:.4f}")

        # === Early Stopping & Save Best ===
        if metrics["AUC"] > best_auc:
            best_auc = metrics["AUC"]
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "metrics": metrics
            }, os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"]))
            logger.info("💾 Best model saved.")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info("⛔ Early stopping triggered.")
                break

    # === Final Save ===
    final_model_path = os.path.join(CONFIG["checkpoint_dir"], "pd_bert_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Final PD BERT model saved at {final_model_path}")

    # === Metrics Summary ===
    logger.info(f"📈 Best AUC: {best_auc:.4f}")

    # === Loss Curve Plot ===
    plt.figure(figsize=(10, 6))
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_auc"], label="Val AUC")
    plt.xlabel("Epochs")
    plt.ylabel("Loss / AUC")
    plt.title("PD BERT Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["checkpoint_dir"], "pd_bert_training_plot.png"))
    logger.info("📉 Training plot saved.")

if __name__ == "__main__":
    train_pd_bert()
