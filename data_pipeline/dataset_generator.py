import os
import json
import logging
from tqdm import tqdm
from typing import List, Dict, Any

from data_pipeline.data_ingestion import load_salary_json, load_credit_json, get_shared_pinfls
from data_pipeline.json_normalizer import normalize_salary_json, normalize_credit_json
from data_pipeline.feature_extractor import FeatureExtractor
from data_pipeline.sequence_extractor import SequenceExtractor

try:
    from services.fraud_service import FraudService
    from services.segment_service import SegmentService
    from services.limit_decision_service import LimitDecisionService
    from models.pd_estimator import PDEstimator
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# === Config ===
OUTPUT_DIR = "./data/processed_datasets/"
MAX_SEQUENCE_LENGTH = 24
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dataset_generator")


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def generate_datasets(pinfl_list: List[str], training_mode: bool = True):
    pd_dataset, fraud_dataset, segment_dataset, limit_dataset = [], [], [], []

    logger.info(f"✅ Starting dataset generation for {len(pinfl_list)} PINFLs...")

    if MODELS_AVAILABLE and not training_mode:
        logger.info("✅ Initializing model services...")
        fraud_service = FraudService()
        segment_service = SegmentService()
        limit_service = LimitDecisionService()
        pd_estimator = PDEstimator()
    else:
        logger.info("⚠️ Training mode or models not available — using fallback values.")
        fraud_service = None
        segment_service = None
        limit_service = None
        pd_estimator = None

    for pinfl in tqdm(pinfl_list, desc="Processing"):
        try:
            # === Load and Normalize ===
            raw_salary = load_salary_json(pinfl)
            raw_credit = load_credit_json(pinfl)

            salary = normalize_salary_json(raw_salary)
            credit = normalize_credit_json(raw_credit)

            # === Feature Extraction ===
            tabular_features = FeatureExtractor(salary, credit).extract_all_features()
            sequence_features = SequenceExtractor(salary, credit).extract()

            # === Predict Limit or Fallback ===
            predicted_limit = 1.0
            if limit_service:
                predicted_limit = limit_service.predict_limit(tabular_features)
            else:
                predicted_limit = credit.get("scorring", {}).get("loan_limit", 1.0)

            total_debt = sum(sequence_features.get("debt", []))
            tabular_features["debt_utilization"] = safe_divide(total_debt, predicted_limit)

            # === Fraud Score ===
            fraud_sequence = sequence_features.get("fraud_sequence", [])
            anomaly_score = 0.0
            if fraud_service:
                anomaly_score = fraud_service.score(fraud_sequence)
            tabular_features["anomaly_score"] = anomaly_score
            tabular_features["fraud_flag"] = float(anomaly_score >= 0.5)

            # === Segment ===
            segment_id = "unknown"
            if segment_service:
                segment_id = segment_service.identify_segment(tabular_features, sequence_features)
            tabular_features["segment_id"] = segment_id

            # === PD Score ===
            pd_input = {
                "salary_sequence": sequence_features.get("salary", []),
                "credit_sequence": sequence_features.get("debt", []),
                "pti_sequence": sequence_features.get("payment_to_income", []),
                "repayment_delta": sequence_features.get("repayment", []),
                "debt_delta": sequence_features.get("debt_delta", []),
            }
            pd_score = 0.0
            if pd_estimator:
                pd_score = pd_estimator.predict(pd_input)
            else:
                pd_score = credit.get("scorring", {}).get("pd_label", 0.0)

            # === Assemble Datasets ===
            pd_dataset.append({
                "pinfl": pinfl,
                **pd_input,
                "label": pd_score,
                "anomaly_score": anomaly_score,
                "segment_id": segment_id
            })

            fraud_dataset.append({
                "pinfl": pinfl,
                "fraud_sequence": fraud_sequence,
                "anomaly_score": anomaly_score,
                "fraud_flag": anomaly_score >= 0.5
            })

            segment_dataset.append({
                "pinfl": pinfl,
                "features": tabular_features,
                "segment_id": segment_id
            })

            limit_dataset.append({
                "pinfl": pinfl,
                "features": tabular_features,
                "loan_limit": predicted_limit
            })

        except Exception as e:
            logger.error(f"❌ Error processing PINFL {pinfl}: {e}")

    # === Save ===
    def save_json(data: List[Dict[str, Any]], filename: str):
        path = os.path.join(OUTPUT_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Saved {filename} ({len(data)} records)")

    save_json(pd_dataset, "pd_dataset.json")
    save_json(fraud_dataset, "fraud_dataset.json")
    save_json(segment_dataset, "segment_dataset.json")
    save_json(limit_dataset, "limit_dataset.json")

    logger.info("✅ All dataset generation completed successfully.")


if __name__ == "__main__":
    pinfl_list = get_shared_pinfls()

# === Limit to 10 PINFLs for testing ===
    LIMIT = 10
    if len(pinfl_list) > LIMIT:
        logger.info(f"⚠️ Limiting dataset generation to {LIMIT} PINFLs (out of {len(pinfl_list)} total).")
        pinfl_list = pinfl_list[:LIMIT]

    generate_datasets(pinfl_list, training_mode=True)
