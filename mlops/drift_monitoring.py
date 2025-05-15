# mlops/drift_monitoring.py

import torch
import numpy as np
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Constants
PSI_THRESHOLD = 0.2  # Above this value = significant drift
KS_THRESHOLD = 0.1   # KS statistic > 0.1 = potential drift

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for a feature.
    """
    expected_percents, _ = np.histogram(expected, bins=bins, range=(np.min(expected), np.max(expected)), density=True)
    actual_percents, _ = np.histogram(actual, bins=bins, range=(np.min(expected), np.max(expected)), density=True)

    expected_percents += 1e-6
    actual_percents += 1e-6

    psi = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
    return psi

def check_input_drift(
    reference_data: torch.Tensor = None,
    current_data: torch.Tensor = None,
    feature_names: list = None
) -> dict:
    """
    Check input feature drift between reference (training) and current (production) data.

    Args:
      reference_data: Tensor (n_samples_ref, n_features)
      current_data: Tensor (n_samples_cur, n_features)
      feature_names: list of feature names (optional)

    Returns:
      drift_report: {feature_name: {psi, ks_stat, drift_detected}}
    """

    logger.info("Checking input drift...")

    if reference_data is None or current_data is None:
        logger.warning("Reference or current data not provided. Skipping drift detection.")
        return {}

    reference_data = reference_data.cpu().numpy()
    current_data = current_data.cpu().numpy()

    n_features = reference_data.shape[1]
    drift_report = {}

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    for i in range(n_features):
        ref_feature = reference_data[:, i]
        cur_feature = current_data[:, i]

        # Calculate PSI
        psi_value = calculate_psi(ref_feature, cur_feature)

        # Calculate KS Test
        ks_stat, ks_pvalue = ks_2samp(ref_feature, cur_feature)

        drift_detected = psi_value > PSI_THRESHOLD or ks_stat > KS_THRESHOLD

        drift_report[feature_names[i]] = {
            "psi": float(psi_value),
            "ks_stat": float(ks_stat),
            "drift_detected": drift_detected
        }

    return drift_report

if __name__ == "__main__":
    # Example dummy test
    torch.manual_seed(42)

    # Reference = normal training features
    reference_data = torch.randn(1000, 10)

    # Current = production features (slightly shifted)
    current_data = reference_data + torch.randn(1000, 10) * 0.1

    report = check_input_drift(reference_data, current_data)

    for feat, result in report.items():
        print(f"{feat}: PSI={result['psi']:.4f}, KS={result['ks_stat']:.4f}, Drift={result['drift_detected']}")
