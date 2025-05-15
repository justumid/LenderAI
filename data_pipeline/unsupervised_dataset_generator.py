# data_pipeline/unsupervised_dataset_generator.py

import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any

from data_ingestion import load_salary_json, load_credit_json
from data_pipeline.json_normalizer import normalize_salary_json, normalize_credit_json
from data_pipeline.sequence_extractor import SequenceExtractor

# === Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unsupervised_dataset_generator")

# === Paths ===
PINFL_LIST_PATH = "./data/pinfl_list.csv"
RAW_SALARY_DIR = "./data/raw/salary/"
RAW_CREDIT_DIR = "./data/raw/credit/"
OUTPUT_DIR = "./data/processed_unsupervised/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Config ===
MAX_SEQUENCE_LENGTH = 24

def load_pinfl_list(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PINFL list: {path}")
    df = pd.read_csv(path)
    return df["pinfl"].astype(str).tolist()

def generate_sequence_dataset(pinfl_list: List[str]) -> List[Dict[str, Any]]:
    dataset = []

    for pinfl in pinfl_list:
        try:
            logger.info(f"🔄 Processing PINFL: {pinfl}")
            raw_salary = load_salary_json(pinfl, RAW_SALARY_DIR)
            raw_credit = load_credit_json(pinfl, RAW_CREDIT_DIR)

            salary = normalize_salary_json(raw_salary)
            credit = normalize_credit_json(raw_credit)

            extractor = SequenceExtractor(salary, credit)

            record = {
                "pinfl": pinfl,
                "salary_sequence": extractor.get_salary_sequence(MAX_SEQUENCE_LENGTH),
                "salary_tax_sequence": extractor.get_salary_tax_sequence(MAX_SEQUENCE_LENGTH),
                "debt_sequence": extractor.get_debt_sequence(MAX_SEQUENCE_LENGTH),
                "debt_delta_sequence": extractor.get_debt_delta_sequence(MAX_SEQUENCE_LENGTH),
                "overdue_days_sequence": extractor.get_overdue_days_sequence(MAX_SEQUENCE_LENGTH),
                "overdue_flag_sequence": extractor.get_overdue_flag_sequence(MAX_SEQUENCE_LENGTH),
                "repayment_sequence": extractor.get_repayment_sequence(MAX_SEQUENCE_LENGTH),
                "payment_to_income_sequence": extractor.get_payment_to_income_sequence(MAX_SEQUENCE_LENGTH),
            }

            dataset.append(record)

        except Exception as e:
            logger.warning(f"⚠️ Failed to process PINFL {pinfl}: {e}")

    return dataset

def save_dataset(data: List[Dict[str, Any]], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Saved unsupervised dataset to: {output_file}")

# === Entry Point ===
if __name__ == "__main__":
    pinfl_list = load_pinfl_list(PINFL_LIST_PATH)
    final_dataset = generate_sequence_dataset(pinfl_list)
    save_dataset(final_dataset, os.path.join(OUTPUT_DIR, "unsupervised_sequences.json"))
    logger.info("✅ Dataset generation complete.")
