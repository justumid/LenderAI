from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging
import subprocess
import os

logger = logging.getLogger(__name__)
router = APIRouter()

# === Request Schema ===
class TrainRequest(BaseModel):
    dataset_path: str = "./demo_data/combined_dataset.json"
    epochs: int = 5
    small_dev: bool = False

class TrainResponse(BaseModel):
    message: str
    status: str

# === Background Training Function ===
def run_training_pipeline(dataset_path: str, epochs: int, small_dev: bool):
    logger.info(f"🚀 Starting Training Pipeline: dataset={dataset_path}, epochs={epochs}, small_dev={small_dev}")

    try:
        command = [
            "python", "train_pipeline.py",
            "--dataset-path", dataset_path,
            "--epochs", str(epochs)
        ]
        if small_dev:
            command.append("--small-dev")

        # Run training as subprocess
        logger.info(f"🔧 Running command: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"❌ Training failed: {result.stderr}")
        else:
            logger.info(f"✅ Training completed successfully: {result.stdout}")

    except Exception as e:
        logger.exception(f"❌ Training pipeline error: {e}")

# === API Endpoint ===
@router.post("/", response_model=TrainResponse)
def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    background_tasks.add_task(run_training_pipeline, request.dataset_path, request.epochs, request.small_dev)

    return TrainResponse(
        message="Training pipeline has been triggered in background.",
        status="started"
    )
