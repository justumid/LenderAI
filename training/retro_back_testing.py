# training/retro_back_testing.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, mean_squared_error
import os
import logging
from tqdm import tqdm

from models.full_model import FullModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = "checkpoints/"

def run_retro_backtesting(
    dataset,
    input_dim: int = 32,
    static_score_dim: int = 1,
    additional_features_dim: int = 0,
    encoder_hidden_dim: int = 128,
    batch_size: int = 64,
    checkpoint_path: str = os.path.join(CHECKPOINT_DIR, "full_model.pt"),
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Simulate historical model scoring and evaluate.
    """

    logger.info("Starting retro backtesting...")

    model = FullModel(
        input_dim=input_dim,
        static_score_dim=static_score_dim,
        additional_features_dim=additional_features_dim,
        encoder_hidden_dim=encoder_hidden_dim
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = {
        "pd": [],
        "fraud": [],
        "loan_limit": []
    }
    all_labels = {
        "pd": [],
        "fraud": [],
        "loan_limit": []
    }

    with torch.no_grad():
        for batch in tqdm(loader, desc="Backtesting", leave=False):
            inputs = batch["x_seq"].to(device)
            static_score = batch["static_score"].to(device)

            additional_features = batch.get("additional_features")
            if additional_features is not None:
                additional_features = additional_features.to(device)

            outputs = model(inputs, static_score, additional_features)

            all_preds["pd"].append(outputs["pd"].cpu())
            all_preds["fraud"].append(outputs["fraud"].cpu())
            all_preds["loan_limit"].append(outputs["loan_limit"].cpu())

            all_labels["pd"].append(batch["pd"].cpu())
            all_labels["fraud"].append(batch["fraud"].cpu())
            all_labels["loan_limit"].append(batch["loan_limit"].cpu())

    # Concatenate all
    for k in all_preds:
        all_preds[k] = torch.cat(all_preds[k]).numpy()
        all_labels[k] = torch.cat(all_labels[k]).numpy()

    # Metrics
    pd_auc = roc_auc_score(all_labels["pd"], all_preds["pd"])
    fraud_auc = roc_auc_score(all_labels["fraud"], all_preds["fraud"])
    limit_rmse = mean_squared_error(all_labels["loan_limit"], all_preds["loan_limit"], squared=False)

    logger.info(f"Retro Backtesting Results:")
    logger.info(f"  - PD AUC: {pd_auc:.4f}")
    logger.info(f"  - Fraud AUC: {fraud_auc:.4f}")
    logger.info(f"  - Loan Limit RMSE: {limit_rmse:.2f}")

    return {
        "pd_auc": pd_auc,
        "fraud_auc": fraud_auc,
        "loan_limit_rmse": limit_rmse
    }

if __name__ == "__main__":
    # Dummy example (replace with real dataset)
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 500

        def __getitem__(self, idx):
            return {
                "x_seq": torch.randn(10, 32),
                "static_score": torch.randn(1),
                "additional_features": torch.randn(5),
                "pd": torch.randint(0, 2, (1,), dtype=torch.float32),
                "fraud": torch.randint(0, 2, (1,), dtype=torch.float32),
                "loan_limit": torch.rand(1) * 1e8
            }

    dataset = DummyDataset()
    run_retro_backtesting(dataset)
