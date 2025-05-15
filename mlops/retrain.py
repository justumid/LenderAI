import logging
from datetime import datetime

# Use train() from your real pipeline file
from training.train_pipeline import train

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info(f"[{datetime.now()}] 🔄 Retraining process started...")

    try:
        train()
        logging.info(f"[{datetime.now()}] ✅ Retraining completed successfully.")
    except Exception as e:
        logging.error(f"[{datetime.now()}] ❌ Retraining failed: {e}")
