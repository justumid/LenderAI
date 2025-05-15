import os
import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

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
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "early_stopping_patience": 5,
    "max_sequence_length": 128
}

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# === Seed for reproducibility ===
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
random.seed(CONFIG["seed"])
if CONFIG["device"] == "cuda":
    torch.cuda.manual_seed_all(CONFIG["seed"])

os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

# === Dataset Class for PD BERT ===
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

# === Training Function ===
def train_pd_bert():
    logger.info("🚀 Starting PD BERT Training Pipeline...")

    # Load or regenerate dataset
    if not os.path.exists(CONFIG["dataset_path"]):
        logger.info(f"⚠️ Dataset not found at {CONFIG['dataset_path']}, regenerating...")
        pd_data = generate_pd_dataset()
        with open(CONFIG["dataset_path"], "w") as f:
            json.dump(pd_data, f, indent=2)
    else:
        logger.info(f"✅ Loading dataset from {CONFIG['dataset_path']}")
        with open(CONFIG["dataset_path"], "r") as f:
            pd_data = json.load(f)

    logger.info(f"✅ Loaded {len(pd_data)} samples.")

    # Prepare DataLoader
    dataset = PDDataset(pd_data)
    dataloader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # Model Initialization
    input_dim = dataset[0][0].shape[0]
    model = PDSequenceClassifier(input_dim=input_dim).to(CONFIG["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    patience_counter = 0
    train_losses = []
    auc_scores = []

    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_loss = 0.0
        all_labels = []
        all_preds = []

        for sequences, labels in dataloader:
            sequences, labels = sequences.to(CONFIG["device"]), labels.to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(sequences).squeeze()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)

        auc = roc_auc_score(all_labels, all_preds)
        auc_scores.append(auc)

        logger.info(f"📊 Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {avg_loss:.6f} - AUC: {auc:.4f}")

        # Early Stopping
        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG["checkpoint_dir"], CONFIG["checkpoint_file"]))
            logger.info(f"💾 Best model saved at {CONFIG['checkpoint_file']}")
        else:
            patience_counter += 1
            logger.info(f"⏳ EarlyStopping Counter: {patience_counter}/{CONFIG['early_stopping_patience']}")
            if patience_counter >= CONFIG["early_stopping_patience"]:
                logger.info(f"⛔ Early stopping triggered after {epoch+1} epochs.")
                break

    # Final Save
    final_model_path = os.path.join(CONFIG["checkpoint_dir"], "pd_bert_final.pt")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"✅ Final PD BERT model saved at {final_model_path}")

    # Metrics Report
    preds_bin = (np.array(all_preds) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_bin)
    f1 = f1_score(all_labels, preds_bin)

    logger.info(f"📈 Final Metrics - AUC: {best_auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    train_pd_bert()
