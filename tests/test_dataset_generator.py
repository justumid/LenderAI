import os
import json
import pprint
from data_pipeline.data_ingestion import load_salary_json, load_credit_json
from data_pipeline.json_normalizer import normalize_salary_json, normalize_credit_json
from data_pipeline.feature_extractor import FeatureExtractor
from data_pipeline.sequence_extractor import SequenceExtractor
from data_pipeline.dataset_generator import OUTPUT_DIR, MAX_SEQUENCE_LENGTH, safe_divide

def test_dataset_generation(pinfl: str):
    print(f"\n🔍 Generating dataset entries for PINFL: {pinfl}")

    # === Load raw JSON from DB ===
    raw_salary = load_salary_json(pinfl)
    raw_credit = load_credit_json(pinfl)

    # === Normalize ===
    norm_salary = normalize_salary_json(raw_salary)
    norm_credit = normalize_credit_json(raw_credit)

    # === Extract Features & Sequences ===
    features = FeatureExtractor(norm_salary, norm_credit).extract_all_features()
    sequences = SequenceExtractor(norm_salary, norm_credit).extract()

    # === Derived Computations ===
    predicted_limit = norm_credit.get("scorring", {}).get("loan_limit", 1.0)
    total_debt = sum(sequences.get("debt", []))
    features["debt_utilization"] = safe_divide(total_debt, predicted_limit)

    # === Fraud & Anomaly placeholders ===
    anomaly_score = 0.0
    features["anomaly_score"] = anomaly_score
    features["fraud_flag"] = float(anomaly_score >= 0.5)

    # === Segment placeholder ===
    segment_id = "unknown"
    features["segment_id"] = segment_id

    # === PD Estimation placeholder ===
    pd_score = norm_credit.get("scorring", {}).get("pd_label", 0.0)

    # === Build Dataset Entries ===
    pd_entry = {
        "pinfl": pinfl,
        "salary_sequence": sequences.get("salary", []),
        "credit_sequence": sequences.get("debt", []),
        "pti_sequence": sequences.get("payment_to_income", []),
        "repayment_delta": sequences.get("repayment", []),
        "debt_delta": sequences.get("debt_delta", []),
        "label": pd_score,
        "anomaly_score": anomaly_score,
        "segment_id": segment_id
    }

    fraud_entry = {
        "pinfl": pinfl,
        "fraud_sequence": sequences.get("fraud_sequence", []),
        "anomaly_score": anomaly_score,
        "fraud_flag": anomaly_score >= 0.5
    }

    segment_entry = {
        "pinfl": pinfl,
        "features": features,
        "segment_id": segment_id
    }

    limit_entry = {
        "pinfl": pinfl,
        "features": features,
        "loan_limit": predicted_limit
    }

    # === Pretty Print Results ===
    print("\n📊 PD Dataset Entry:")
    pprint.pprint(pd_entry)

    print("\n📊 Fraud Dataset Entry:")
    pprint.pprint(fraud_entry)

    print("\n📊 Segment Dataset Entry:")
    pprint.pprint(segment_entry)

    print("\n📊 Limit Dataset Entry:")
    pprint.pprint(limit_entry)

    # === Optional: Save outputs ===
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"pd_dataset_{pinfl}.json"), "w") as f:
        json.dump([pd_entry], f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, f"fraud_dataset_{pinfl}.json"), "w") as f:
        json.dump([fraud_entry], f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, f"segment_dataset_{pinfl}.json"), "w") as f:
        json.dump([segment_entry], f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, f"limit_dataset_{pinfl}.json"), "w") as f:
        json.dump([limit_entry], f, indent=2, ensure_ascii=False)

    print(f"\n✅ Test datasets saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    test_pinfl = "30303901621130"  # Replace with a real PINFL from DB
    test_dataset_generation(test_pinfl)
