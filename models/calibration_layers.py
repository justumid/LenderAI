import torch
import torch.nn as nn
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from scipy.special import expit  # sigmoid function
from typing import Dict, Any

# === Temperature Scaling (as Platt Scaling proxy) ===
class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for binary classifiers (Platt Scaling-like).
    """

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.training:
            return logits / self.temperature
        else:
            return logits / self.temperature.detach()

    def calibrate(self, logits: torch.Tensor, labels: torch.Tensor, loss_fn=nn.BCEWithLogitsLoss()):
        """
        Optimize temperature parameter on validation set.
        """
        self.train()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()
        print(f"✅ Calibrated temperature: {self.temperature.item():.4f}")

# === Classic Platt Scaling ===
class PlattScaler:
    """
    Platt Scaling via logistic regression (scikit-learn).
    """

    def __init__(self):
        self.model = None

    def fit(self, logits: np.ndarray, targets: np.ndarray):
        """
        Fit logistic regression to map logits to probabilities.
        """
        logits = logits.reshape(-1, 1)  # Ensure 2D
        self.model = LogisticRegression(solver='lbfgs')
        self.model.fit(logits, targets)
        print("✅ PlattScaler calibrated.")

    def predict(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to logits.
        """
        if self.model is None:
            raise ValueError("PlattScaler is not fitted yet.")
        logits = logits.reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]

    def score(self, logits: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute AUC and MAE after calibration.
        """
        calibrated = self.predict(logits)
        auc = roc_auc_score(targets, calibrated)
        mae = mean_absolute_error(targets, calibrated)
        print(f"✅ PlattScaler AUC: {auc:.4f} | MAE: {mae:.4f}")
        return {"auc": auc, "mae": mae}

# === Isotonic Regression Calibration ===
class IsotonicCalibrator:
    """
    Isotonic Regression calibration (non-parametric).
    """

    def __init__(self):
        self.model = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        self.model = IsotonicRegression(out_of_bounds='clip')
        self.model.fit(y_pred, y_true)
        print("✅ Isotonic regression calibrated.")

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("IsotonicCalibrator is not fitted yet.")
        return self.model.predict(y_pred)

    def score(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        y_calibrated = self.predict(y_pred)
        auc = roc_auc_score(y_true, y_calibrated)
        mae = mean_absolute_error(y_true, y_calibrated)
        print(f"✅ IsotonicCalibrator AUC: {auc:.4f} | MAE: {mae:.4f}")
        return {"auc": auc, "mae": mae}
