# training/train_epoch.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(
    model: nn.Module,
    dataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    batch_size: int = 64
) -> float:
    """
    Train model for one epoch.
    """
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        inputs = batch["x_seq"].to(device)
        static_score = batch["static_score"].to(device)

        additional_features = batch.get("additional_features")
        if additional_features is not None:
            additional_features = additional_features.to(device)

        outputs = model(inputs, static_score, additional_features)

        # Build targets dictionary
        targets = {
            "pd": batch["pd"].to(device),
            "ead": batch["ead"].to(device),
            "lgd": batch["lgd"].to(device),
            "fraud": batch["fraud"].to(device),
            "loan_limit": batch["loan_limit"].to(device)
        }

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss

def validate_epoch(
    model: nn.Module,
    dataset,
    criterion: nn.Module,
    device: str,
    batch_size: int = 64
) -> float:
    """
    Validate model for one epoch.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            inputs = batch["x_seq"].to(device)
            static_score = batch["static_score"].to(device)

            additional_features = batch.get("additional_features")
            if additional_features is not None:
                additional_features = additional_features.to(device)

            outputs = model(inputs, static_score, additional_features)

            targets = {
                "pd": batch["pd"].to(device),
                "ead": batch["ead"].to(device),
                "lgd": batch["lgd"].to(device),
                "fraud": batch["fraud"].to(device),
                "loan_limit": batch["loan_limit"].to(device)
            }

            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss
