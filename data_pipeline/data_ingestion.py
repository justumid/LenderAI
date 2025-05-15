# data_pipeline/data_ingestion.py

import os
import json
import psycopg2
from typing import Optional, Dict, Any
from psycopg2.extras import RealDictCursor
import logging

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Output directory for raw dumps
OUTPUT_DIR = "demo_data/raw_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Database configuration
DB_CONFIG = {
    "dbname": "credit_scoring",
    "user": "postgres",
    "password": "java2006",
    "host": "localhost",
    "port": 5432
}

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_salary_json(pinfl: str) -> Optional[Dict[str, Any]]:
    query = """
        SELECT salary_data
        FROM salary_records
        WHERE pinfl = %s
        LIMIT 1;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (pinfl,))
                result = cursor.fetchone()

        if not result or not result.get("salary_data"):
            print(f"[DEBUG] No salary_data found for PINFL {pinfl}")
            return None

        data = result["salary_data"]
        print(f"[DEBUG] Raw salary JSON for {pinfl}:", type(data), str(data)[:200])

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                logger.warning(f"[Salary JSON] Failed to parse string: {e}")
                return None

        if isinstance(data, list):
            print(f"[DEBUG] salary_data is a list for PINFL {pinfl} — wrapping in dict")
            parsed = {"salary_data": data}
        elif isinstance(data, dict):
            parsed = data
        else:
            print(f"[DEBUG] salary_data is neither dict nor list for PINFL {pinfl}")
            return None

        with open(os.path.join(OUTPUT_DIR, f"salary_{pinfl}.json"), "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        return parsed

    except Exception as e:
        print(f"[ERROR] Salary load failed for PINFL {pinfl}: {e}")
        return None


def load_credit_json(pinfl: str) -> Optional[Dict]:
    query = """
        SELECT credit_data
        FROM credit_records
        WHERE "pPinfl" = %s
        LIMIT 1;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (pinfl,))
                result = cursor.fetchone()

        if not result or not result.get("credit_data"):
            print(f"[DEBUG] No credit_data found for PINFL {pinfl}")
            return None

        data = result["credit_data"]
        print(f"[DEBUG] Raw credit JSON (1st level) for {pinfl}:", type(data), str(data)[:200])

        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception as e:
                logger.warning(f"[load_credit_json] Failed to parse JSON string: {e}")
                return None

        if isinstance(data, list):
            parsed = data[0] if data else {}
        elif isinstance(data, dict):
            parsed = data
        else:
            print(f"[DEBUG] credit_data is neither dict nor list for PINFL {pinfl}")
            return None

        with open(os.path.join(OUTPUT_DIR, f"credit_{pinfl}.json"), "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        return parsed

    except Exception as e:
        logger.error(f"[load_credit_json] DB Error: {e}")
        return None


def get_shared_pinfls(limit: Optional[int] = None) -> list[str]:
    query = """
        SELECT DISTINCT s.pinfl
        FROM salary_records s
        INNER JOIN credit_records c ON s.pinfl = c."pPinfl"
        WHERE s.salary_data IS NOT NULL AND c.credit_data IS NOT NULL
    """
    if limit:
        query += f" LIMIT {limit}"

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                return [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"[DB] Failed to fetch shared PINFLs: {e}")
        return []


if __name__ == "__main__":
    test_pinfl = "30202836860013"
    print(f"[TEST] Fetching raw data for PINFL {test_pinfl}")

    salary = load_salary_json(test_pinfl)
    credit = load_credit_json(test_pinfl)

    if salary:
        print(f"[TEST] ✅ Salary JSON saved for {test_pinfl}")
    if credit:
        print(f"[TEST] ✅ Credit JSON saved for {test_pinfl}")
