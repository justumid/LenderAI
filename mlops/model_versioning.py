# mlops/model_versioning.py

import os
import json
import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MODEL_VERSIONS_DIR = "model_versions/"
os.makedirs(MODEL_VERSIONS_DIR, exist_ok=True)

def register_new_model_version(version_id: str, notes: str = ""):
    """
    Register a new model version with metadata.

    Args:
      version_id: version identifier (e.g., "v20240501_1500")
      notes: optional notes (e.g., "retrained after drift")
    """

    metadata = {
        "version_id": version_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "notes": notes
    }

    metadata_path = os.path.join(MODEL_VERSIONS_DIR, f"{version_id}.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model version {version_id} registered successfully.")

def list_all_model_versions():
    """
    List all registered model versions.
    """

    files = os.listdir(MODEL_VERSIONS_DIR)
    version_files = [f for f in files if f.endswith(".json")]

    if not version_files:
        logger.info("No model versions registered yet.")
        return []

    versions = []
    for vf in version_files:
        with open(os.path.join(MODEL_VERSIONS_DIR, vf), "r") as f:
            metadata = json.load(f)
            versions.append(metadata)

    versions = sorted(versions, key=lambda x: x["timestamp"])
    return versions

if __name__ == "__main__":
    # Example
    new_version = datetime.datetime.now().strftime("v%Y%m%d_%H%M%S")
    register_new_model_version(new_version, notes="First full training.")

    all_versions = list_all_model_versions()
    for v in all_versions:
        print(f"Version {v['version_id']} at {v['timestamp']} | Notes: {v['notes']}")
