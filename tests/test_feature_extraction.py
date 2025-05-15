import os
import json
import pprint
from data_pipeline.data_ingestion import load_salary_json, load_credit_json
from data_pipeline.json_normalizer import normalize_salary_json, normalize_credit_json
from data_pipeline.feature_extractor import FeatureExtractor

def test_feature_extraction(pinfl: str, save_output: bool = True):
    print(f"\n🔍 Extracting features for PINFL: {pinfl}")

    # === Load raw data from DB ===
    raw_salary = load_salary_json(pinfl)
    raw_credit = load_credit_json(pinfl)

    # === Normalize raw data ===
    norm_salary = normalize_salary_json(raw_salary)
    norm_credit = normalize_credit_json(raw_credit)

    # === Extract Features ===
    extractor = FeatureExtractor(norm_salary, norm_credit)
    features = extractor.extract_all_features()

    # === Pretty Print the Extracted Features ===
    print("\n📊 Extracted Features Summary:")
    key_fields = ["salary_mean", "salary_last_3mo_avg", "dti_ratio", "pti_ratio", 
                  "overdue_risk_flag", "scoring_grade", "scoring_class", 
                  "debt_utilization", "safe_payment_capability"]
    for field in key_fields:
        print(f"  {field}: {features.get(field)}")

    print("\n📊 Full Feature Set:")
    pprint.pprint(features)

    # === Optional: Save full features to JSON ===
    if save_output:
        os.makedirs("demo_data", exist_ok=True)
        with open(f"demo_data/features_{pinfl}.json", "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Features saved to demo_data/features_{pinfl}.json")

if __name__ == "__main__":
    # === Replace this with any valid PINFL from your DB ===
    test_pinfl = "30303901621130"
    test_feature_extraction(test_pinfl)
