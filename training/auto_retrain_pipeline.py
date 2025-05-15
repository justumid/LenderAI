# training/auto_retrain_pipeline.py

import logging
from mlops.drift_monitoring import check_input_drift
from mlops.calibration_drift import check_calibration_drift
from training.train_pipeline import run_full_training_pipeline
from mlops.auto_deployment import auto_deploy_best_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DRIFT_FEATURE_THRESHOLD = 5  # Number of features drifted to trigger retraining
CALIBRATION_SHIFT_THRESHOLD = 0.03  # Calibration drift threshold

def auto_retrain_if_needed(dataset):
    """
    Automatically check for drift, retrain model if necessary.
    """

    logger.info("Auto-retraining check triggered...")

    input_drift = check_input_drift()
    calibration_drift = check_calibration_drift()

    # Count number of drifted features
    drifted_features = [f for f, result in input_drift.items() if result["drift_detected"]]
    num_drifted = len(drifted_features)
    calibration_drift_detected = calibration_drift.get("drift_detected", False)

    logger.info(f"Detected {num_drifted} features with input drift.")
    logger.info(f"Calibration drift detected: {calibration_drift_detected}")

    retrain_needed = False

    if num_drifted >= DRIFT_FEATURE_THRESHOLD:
        logger.warning(f"Input drift exceeds threshold ({num_drifted} features). Retraining needed.")
        retrain_needed = True

    if calibration_drift_detected:
        logger.warning(f"Calibration drift detected. Retraining needed.")
        retrain_needed = True

    if retrain_needed:
        logger.info("Starting full retraining...")
        run_full_training_pipeline(dataset=dataset)
        logger.info("Retraining completed.")

        logger.info("Starting auto deployment of new model...")
        auto_deploy_best_model()
        logger.info("Auto deployment completed.")

    else:
        logger.info("No retraining needed. Model is stable.")

if __name__ == "__main__":
    # Dummy example
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1000

        def __getitem__(self, idx):
            return {
                "x_seq": torch.randn(10, 32),
                "static_score": torch.randn(1),
                "additional_features": torch.randn(5),
                "pd": torch.randint(0, 2, (1,), dtype=torch.float32),
                "ead": torch.rand(1),
                "lgd": torch.rand(1),
                "fraud": torch.randint(0, 2, (1,), dtype=torch.float32),
                "loan_limit": torch.rand(1) * 1e8
            }

    dataset = DummyDataset()
    auto_retrain_if_needed(dataset)
