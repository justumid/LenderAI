import logging
from typing import Dict, Any

from lenderai.data_pipeline.feature_extractor import FeatureExtractor
from lenderai.services.static_scoring_engine import StaticScoringEngine
from lenderai.services.fraud_service import detect_fraud
from lenderai.services.limit_decision_service import predict_limit
from lenderai.services.segment_service import identify_segment
from lenderai.services.risk_engine import predict_risk
from lenderai.services.vae_anomaly_engine import compute_anomaly_score
from lenderai.services.explanation_service import generate_explanations
from lenderai.services.identity_service import get_identity_info
from lenderai.services.preprocessing import preprocess_applicant, PreprocessingError

# === Logger ===
logger = logging.getLogger(__name__)
scoring_engine = StaticScoringEngine("lenderai/configs/static_rules.yaml")

# === Risk Matrix Definition ===
RISK_MATRIX = {
    "low": {"min": 70, "max": 100},
    "medium": {"min": 40, "max": 70},
    "high": {"min": 0, "max": 40}
}

# === Helper: Classify Risk based on PD Score ===
def classify_risk(pd_score: float) -> str:
    for risk_class, bounds in RISK_MATRIX.items():
        if bounds["min"] <= pd_score < bounds["max"]:
            return risk_class
    return "unknown"

# === Helper: Calibrate Loan Limit based on segment & anomaly ===
def calibrate_limit(base_limit: float, segment_id: int, anomaly_score: float) -> float:
    segment_penalty = 1.0 - segment_id * 0.02  # 2% per segment
    anomaly_penalty = 1.0 - min(anomaly_score / 10, 0.1)  # max 10% penalty
    adjusted_limit = base_limit * segment_penalty * anomaly_penalty
    return max(adjusted_limit, 0.0)

# === Main Label Generator ===
def generate_labels(pinfl: str) -> Dict[str, Any]:
    try:
        # === Step 1: Preprocess Applicant Data ===
        tensors = preprocess_applicant(pinfl)
        if not tensors:
            raise PreprocessingError("Preprocessing failed for PINFL.")

        raw_salary = tensors["raw_salary"]
        raw_credit = tensors["raw_credit"]
        x_seq = tensors["x_seq"]
        additional = tensors["additional_features"]

        # === Step 2: Feature Extraction ===
        extractor = FeatureExtractor(raw_salary, raw_credit)
        tabular_features = extractor.extract_all_features()
        modern_indicators = extractor.extract_modern_indicators()

        logger.info(f"[PINFL={pinfl}] ✅ Features extracted.")

        # === Step 3: Static Scoring ===
        static_result = scoring_engine.compute_score({**tabular_features, **modern_indicators})
        static_score = static_result["score"]
        auto_reject = static_result.get("auto_reject", False)
        stop_remark = static_result.get("stop_remark", "")

        if auto_reject:
            logger.warning(f"[PINFL={pinfl}] ❌ Auto-reject triggered: {stop_remark}")
            return {
                "pinfl": pinfl,
                "pd_score": 0.0,
                "loan_limit": 0.0,
                "fraud_flag": True,
                "risk_class": "high",
                "confidence": {"pd": 1.0, "fraud": 1.0, "risk": 1.0},
                "explanations": [],
                "audit_trail": {"auto_reject_reason": stop_remark},
                "auto_reject": True
            }

        # === Step 4: Model Predictions ===
        segment_id = identify_segment(x_seq)
        fraud_score, fraud_flag = detect_fraud(x_seq, additional, static_score)
        risk_score = predict_risk(x_seq, additional, static_score)
        raw_limit = predict_limit(x_seq, additional, static_score)
        vae_anomaly = compute_anomaly_score(x_seq)

        logger.info(f"[PINFL={pinfl}] Fraud={fraud_score}, Risk={risk_score}, Limit={raw_limit}")

        # === Step 5: PD Score Fusion ===
        pd_score = round(
            0.4 * static_score +
            0.3 * (1 - fraud_score) * 100 +
            0.3 * (1 - risk_score) * 100, 2
        )

        # === Step 6: Risk Class Determination ===
        risk_class = classify_risk(pd_score)

        # === Step 7: Limit Calibration ===
        final_limit = calibrate_limit(raw_limit, segment_id, vae_anomaly)

        # === Step 8: Confidence Scores ===
        confidence = {
            "pd": round(1 - abs(pd_score - static_score) / 100, 3),
            "fraud": round(1 - fraud_score, 3),
            "risk": round(1 - risk_score, 3)
        }

        # === Step 9: Explanations ===
        explanations = generate_explanations(static_score, risk_score, fraud_score, final_limit)

        # === Step 10: Identity Info ===
        identity = get_identity_info(raw_salary, raw_credit)

        # === Final Result Dictionary ===
        result = {
            "pinfl": pinfl,
            "pd_score": pd_score,
            "loan_limit": final_limit,
            "fraud_flag": fraud_flag,
            "risk_class": risk_class,
            "confidence": confidence,
            "segment_id": segment_id,
            "vae_anomaly": vae_anomaly,
            "explanations": explanations,
            "identity_info": identity,
            "audit_trail": {
                "static_score": static_score,
                "fraud_score": fraud_score,
                "risk_score": risk_score,
                "raw_limit": raw_limit,
                "calibrated_limit": final_limit,
                "anomaly_score": vae_anomaly,
                "segment_id": segment_id
            },
            "auto_reject": False
        }

        logger.info(f"[PINFL={pinfl}] ✅ Labels generated successfully.")
        return result

    except PreprocessingError as e:
        logger.error(f"[PINFL={pinfl}] ❌ Preprocessing Error: {e}")
        return {"pinfl": pinfl, "error": str(e)}
