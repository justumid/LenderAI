import os
import json
import pprint
import logging
from data_pipeline.data_ingestion import load_salary_json, load_credit_json
from data_pipeline.json_normalizer import normalize_salary_json, normalize_credit_json
from data_pipeline.sequence_extractor import SequenceExtractor

MAX_SEQUENCE_LENGTH = 24

def test_sequence_extraction(pinfl: str):
    logger = logging.getLogger(__name__)
    logger.info(f"🔍 Extracting sequences for PINFL: {pinfl}")

    # === Load raw data ===
    raw_salary = load_salary_json(pinfl)
    raw_credit = load_credit_json(pinfl)

    if raw_salary is None or raw_credit is None:
        logger.error(f"❌ Failed to load salary or credit data for PINFL {pinfl}. Aborting test.")
        return

    # === Normalize data ===
    norm_salary = normalize_salary_json(raw_salary)
    norm_credit = normalize_credit_json(raw_credit)

    logger.info(f"✅ Data normalized successfully for PINFL: {pinfl}")

    # === Initialize SequenceExtractor ===
    extractor = SequenceExtractor(
        salary_data=norm_salary,
        credit_data=norm_credit,
        default_max_sequence_length=MAX_SEQUENCE_LENGTH,
        current_date_for_timeline=None  # Use system date
    )

    # === Extract Sequences ===
    sequences = extractor.extract(MAX_SEQUENCE_LENGTH)

    # === Pretty Print Results ===
    print("\n📈 Extracted Sequences Summary:")
    for name, seq in sequences.items():
        print(f"\n🔹 {name} (Length: {len(seq)})")
        if isinstance(seq, list):
            if seq and isinstance(seq[0], dict):
                pprint.pprint(seq[0])
            elif seq and isinstance(seq[0], list):
                print(f"  Feature size: {len(seq[0])}, Sample: {seq[0]}")
            else:
                print(f"  Sample values: {seq[:5]}")
        else:
            print(f"  Unexpected type: {type(seq)}")

    # === Save Output to JSON File ===
    os.makedirs("demo_data", exist_ok=True)
    output_path = f"demo_data/sequences_{pinfl}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sequences, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Sequences saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_pinfl = "30303901621130"  # Replace with a valid PINFL
    test_sequence_extraction(test_pinfl)
