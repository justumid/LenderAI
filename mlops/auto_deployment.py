# mlops/auto_deployment.py

import os
import shutil
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHECKPOINT_DIR = "checkpoints/"
CURRENT_MODEL_DIR = "current_model/"
os.makedirs(CURRENT_MODEL_DIR, exist_ok=True)

def auto_deploy_best_model(
    best_model_filename: str = "full_model.pt"
):
    """
    Deploy best model into production folder.

    Args:
      best_model_filename: model checkpoint name (default: 'full_model.pt')
    """

    source_path = os.path.join(CHECKPOINT_DIR, best_model_filename)
    target_path = os.path.join(CURRENT_MODEL_DIR, "full_model.pt")

    if not os.path.exists(source_path):
        logger.error(f"Best model {source_path} does not exist. Cannot deploy.")
        return False

    shutil.copyfile(source_path, target_path)
    logger.info(f"✅ Best model deployed to production: {target_path}")

    # Optional: You can trigger an API reload here (FastAPI reload endpoint etc.)

    return True

if __name__ == "__main__":
    # Dummy deploy test
    success = auto_deploy_best_model()

    if success:
        logger.info("Model deployment completed successfully!")
    else:
        logger.error("Model deployment failed!")
