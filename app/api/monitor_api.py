from fastapi import APIRouter
from pydantic import BaseModel
import logging
import random

from mlops.drift_monitoring import check_drift
from mlops.calibration_drift import check_calibration
from mlops.model_versioning import get_current_model_version

logger = logging.getLogger(__name__)
router = APIRouter()

# === Response Schema ===
class MonitorResponse(BaseModel):
    model_version: str
    drift_status: str
    drift_metric: float
    calibration_status: str
    calibration_metric: float
    model_health: str

# === API Endpoint ===
@router.get("/", response_model=MonitorResponse)
def monitor_status():
    logger.info("🔍 Running model monitoring checks...")

    try:
        # === Model Version ===
        model_version = get_current_model_version()

        # === Drift Check ===
        drift_detected, drift_metric = check_drift()
        drift_status = "drift_detected" if drift_detected else "no_drift"

        # === Calibration Check ===
        calibration_ok, calibration_metric = check_calibration()
        calibration_status = "calibrated" if calibration_ok else "miscalibrated"

        # === Model Health (simple heuristic) ===
        if drift_detected or not calibration_ok:
            model_health = "needs_attention"
        else:
            model_health = "healthy"

        logger.info(f"✅ Monitoring status: {model_health}")

        return MonitorResponse(
            model_version=model_version,
            drift_status=drift_status,
            drift_metric=drift_metric,
            calibration_status=calibration_status,
            calibration_metric=calibration_metric,
            model_health=model_health
        )

    except Exception as e:
        logger.exception(f"❌ Monitoring check failed: {e}")
        # Return dummy fallback
        return MonitorResponse(
            model_version="unknown",
            drift_status="error",
            drift_metric=0.0,
            calibration_status="error",
            calibration_metric=0.0,
            model_health="unknown"
        )
