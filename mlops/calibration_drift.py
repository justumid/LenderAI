# mlops/calibration_drift.py

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
ECE_THRESHOLD = 0.03  # 3% calibration shift allowed

def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    Args:
      probs: predicted probabilities
      labels: true binary labels (0/1)
    Returns:
      ece: expected calibration error
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (probs > bin_lower) & (probs <= bin_upper)
        bin_size = np.sum(mask)

        if bin_size > 0:
            avg_prob = np.mean(probs[mask])
            avg_label = np.mean(labels[mask])
            ece += np.abs(avg_prob - avg_label) * (bin_size / len(probs))

    return ece

def check_calibration_drift(
    predicted_probs: torch.Tensor = None,
    true_labels: torch.Tensor = None
) -> dict:
    """
    Check if model calibration drift occurred.

    Args:
      predicted_probs: Tensor (n_samples,) - model predicted probabilities
      true_labels: Tensor (n_samples,) - ground truth labels

    Returns:
      dict: {"ece": value, "drift_detected": bool}
    """

    logger.info("Checking calibration drift...")

    if predicted_probs is None or true_labels is None:
        logger.warning("Predicted probs or true labels not provided. Skipping calibration drift.")
        return {}

    probs = predicted_probs.detach().cpu().numpy()
    labels = true_labels.detach().cpu().numpy()

    ece_value = expected_calibration_error(probs, labels)

    drift_detected = ece_value > ECE_THRESHOLD

    return {
        "ece": float(ece_value),
        "drift_detected": drift_detected
    }

if __name__ == "__main__":
    # Example dummy test
    torch.manual_seed(42)

    # Dummy good calibrated model
    preds_good = torch.rand(1000)
    labels_good = (preds_good > 0.5).float()

    result_good = check_calibration_drift(preds_good, labels_good)
    print(f"Good Model ECE: {result_good['ece']:.4f}, Drift Detected: {result_good['drift_detected']}")

    # Dummy bad calibrated model
    preds_bad = torch.rand(1000)
    labels_bad = torch.randint(0, 2, (1000,), dtype=torch.float32)

    result_bad = check_calibration_drift(preds_bad, labels_bad)
    print(f"Bad Model ECE: {result_bad['ece']:.4f}, Drift Detected: {result_bad['drift_detected']}")
