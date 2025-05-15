# mlops/auto_rollback.py

import os
import shutil
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CURRENT_MODEL_DIR = "current_model/"
BACKUP_MODEL_DIR = "backup_model/"
os.makedirs(BACKUP_MODEL_DIR, exist_ok=True)

def backup_current_model():
    """
    Before deploying new model, backup the existing production model.
    """

    current_model_path = os.path.join(CURRENT_MODEL_DIR, "full_model.pt")
    backup_model_path = os.path.join(BACKUP_MODEL_DIR, "full_model_backup.pt")

    if os.path.exists(current_model_path):
        shutil.copyfile(current_model_path, backup_model_path)
        logger.info(f"✅ Current model backed up to {backup_model_path}")
    else:
        logger.warning("⚠️ No current model found to backup.")

def rollback_to_previous_model():
    """
    Rollback production model to the last backup.
    """

    backup_model_path = os.path.join(BACKUP_MODEL_DIR, "full_model_backup.pt")
    current_model_path = os.path.join(CURRENT_MODEL_DIR, "full_model.pt")

    if not os.path.exists(backup_model_path):
        logger.error("❌ No backup model available for rollback!")
        return False

    shutil.copyfile(backup_model_path, current_model_path)
    logger.info(f"✅ Rollback completed. Production model restored to previous backup.")

    # Optional: Trigger scoring API reload here if needed.

    return True

if __name__ == "__main__":
    # Dummy test
    backup_current_model()

    # Later if bad deployment happens:
    rollback_to_previous_model()
