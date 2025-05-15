import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import matplotlib.pyplot as plt
import os
import time

class DeepTrainer:
    def __init__(self, models, optimizer, loss_funcs, device, checkpoint_dir="checkpoints", early_stop_patience=5, lr_scheduler=None):
        self.models = models
        self.optimizer = optimizer
        self.loss_funcs = loss_funcs
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.early_stop_patience = early_stop_patience
        self.lr_scheduler = lr_scheduler

        self.best_val_loss = float('inf')
        self.patience_counter = 0

        run_id = time.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir=Path("runs") / run_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, loader, epoch, val_loader=None):
        for model in self.models.values():
            model.train()

        epoch_losses = {k: 0.0 for k in ["recon", "fraud", "risk", "limit", "total"]}
        y_true, y_pred = [], []
        total_batches = len(loader)

        for batch in loader:
            batch_x, static_score, y_risk, y_fraud, y_limit = self._unpack_batch(batch)

            batch_x, static_score = batch_x.to(self.device), static_score.to(self.device)
            y_risk, y_fraud, y_limit = y_risk.to(self.device), y_fraud.to(self.device), y_limit.to(self.device)

            # === Forward Pass ===
            embedding = self.models["encoder"](batch_x[:, :, :5], batch_x[:, :, 5:])
            recon, _, _ = self.models["vae"](batch_x)
            simclr_embedding = self.models["simclr"](batch_x)

            # === Loss Computation ===
            recon_loss = self.loss_funcs["mse"](recon, batch_x)

            fraud_input = torch.cat([simclr_embedding, static_score], dim=1)
            fraud_logits = self.models["fraud_head"](fraud_input)
            fraud_loss = F.binary_cross_entropy(fraud_logits.squeeze(), y_fraud.squeeze())

            risk_input = torch.cat([embedding, static_score], dim=1)
            risk_logits = self.models["risk_head"](risk_input)
            risk_loss = F.binary_cross_entropy(risk_logits.squeeze(), y_risk.squeeze())

            limit_input = torch.cat([embedding, static_score], dim=1)
            limit_pred = self.models["limit_head"](limit_input)
            limit_loss = F.mse_loss(limit_pred.squeeze(), y_limit.squeeze())

            total_loss = recon_loss + fraud_loss + risk_loss + limit_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # === Metrics ===
            y_true.extend(y_fraud.cpu().numpy())
            y_pred.extend(fraud_logits.detach().cpu().numpy())

            # === Accumulate Losses ===
            epoch_losses["recon"] += recon_loss.item()
            epoch_losses["fraud"] += fraud_loss.item()
            epoch_losses["risk"] += risk_loss.item()
            epoch_losses["limit"] += limit_loss.item()
            epoch_losses["total"] += total_loss.item()

        # === Average Loss Logging ===
        for k in epoch_losses:
            epoch_losses[k] /= total_batches
            self.writer.add_scalar(f"Loss/Train_{k}", epoch_losses[k], epoch)

        # === Metrics Logging ===
        self._log_metrics(y_true, y_pred, epoch, prefix="Train")

        # === Validation ===
        if val_loader:
            val_loss, val_y_true, val_y_pred = self.validate(val_loader, epoch)

            if val_loss < self.best_val_loss:
                print(f"✅ New best val_loss {val_loss:.4f} (epoch {epoch}) — Saving checkpoint.")
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_all_models(suffix="best")
            else:
                self.patience_counter += 1
                print(f"EarlyStopping patience: {self.patience_counter}/{self.early_stop_patience}")

            # LR Scheduler Step
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            # Early Stopping
            if self.patience_counter >= self.early_stop_patience:
                print("⏹️ Early stopping triggered.")
                return epoch_losses  # Stop training loop early

            self._log_metrics(val_y_true, val_y_pred, epoch, prefix="Val")

        return epoch_losses

    def validate(self, val_loader, epoch):
        for model in self.models.values():
            model.eval()

        val_losses = []
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch_x, static_score, _, y_fraud, _ = self._unpack_batch(batch)

                batch_x, static_score = batch_x.to(self.device), static_score.to(self.device)
                y_fraud = y_fraud.to(self.device)

                simclr_embedding = self.models["simclr"](batch_x)
                fraud_input = torch.cat([simclr_embedding, static_score], dim=1)
                fraud_logits = self.models["fraud_head"](fraud_input)

                loss = F.binary_cross_entropy(fraud_logits.squeeze(), y_fraud.squeeze())
                val_losses.append(loss.item())

                y_true.extend(y_fraud.cpu().numpy())
                y_pred.extend(fraud_logits.detach().cpu().numpy())

        avg_loss = np.mean(val_losses)
        self.writer.add_scalar("Loss/Val_total", avg_loss, epoch)
        return avg_loss, y_true, y_pred

    def _log_metrics(self, y_true, y_pred, epoch, prefix="Train"):
        try:
            y_pred_bin = np.round(y_pred)
            auc = roc_auc_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred_bin, zero_division=0)
            rec = recall_score(y_true, y_pred_bin, zero_division=0)
            f1 = f1_score(y_true, y_pred_bin, zero_division=0)
            cm = confusion_matrix(y_true, y_pred_bin)

            self.writer.add_scalar(f"Metrics/{prefix}_AUC", auc, epoch)
            self.writer.add_scalar(f"Metrics/{prefix}_Precision", prec, epoch)
            self.writer.add_scalar(f"Metrics/{prefix}_Recall", rec, epoch)
            self.writer.add_scalar(f"Metrics/{prefix}_F1", f1, epoch)

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay(cm).plot(cmap="Blues", ax=ax)
            ax.set_title(f"{prefix} Confusion Matrix — Epoch {epoch}")
            self.writer.add_figure(f"{prefix}/ConfusionMatrix", fig, epoch)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Metric Logging failed: {e}")

    def _save_all_models(self, suffix="best"):
        for name, model in self.models.items():
            torch.save(model.state_dict(), os.path.join(self.checkpoint_dir, f"{name}_{suffix}.pt"))

    def _unpack_batch(self, batch):
        # Handles flexible batch formats
        if len(batch) == 2:
            batch_x, static_score = batch
            y_risk = static_score
            y_fraud = (static_score < 0.5).float()
            y_limit = static_score * 50_000_000
        else:
            batch_x, static_score, y_risk, y_fraud, y_limit = batch
        return batch_x, static_score, y_risk, y_fraud, y_limit
