# data_pipeline/data_ingestion.py

import psycopg2
import json
from typing import Optional, Dict, List
from psycopg2.extras import RealDictCursor

# --- DB Configuration ---
DB_CONFIG = {
    "dbname": "credit_scoring",
    "user": "postgres",
    "password": "java2006",
    "host": "localhost",
    "port": 5432
}

def get_db_connection():
    """Create PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)

def get_shared_pinfls() -> List[str]:
    """
    Get list of PINFLs that exist in both salary_records and credit_records.
    """
    query = """
        SELECT DISTINCT s.pinfl
        FROM salary_records s
        INNER JOIN credit_records c ON s.pinfl = c."pPinfl"
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(query)
    pinfls = [row[0] for row in cur.fetchall()]
    conn.close()
    return pinfls

def load_salary_json(pinfl: str) -> Optional[Dict]:
    """
    Load parsed salary JSON for a specific PINFL.
    """
    query = """
        SELECT salary_data
        FROM salary_records
        WHERE pinfl = %s
        LIMIT 1;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (pinfl,))
        result = cur.fetchone()
        conn.close()

        if not result or not result.get("salary_data"):
            return None

        data = result["salary_data"]
        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, list):
            return {"salary": data}
        elif isinstance(data, dict):
            return data
        else:
            print(f"[Warning] Unexpected salary format for PINFL {pinfl}")
            return None
    except Exception as e:
        print(f"[Error] Failed to load salary for {pinfl}: {e}")
        return None

def load_credit_json(pinfl: str) -> Optional[Dict]:
    """
    Load parsed credit JSON for a specific PINFL.
    """
    query = """
        SELECT credit_data
        FROM credit_records
        WHERE "pPinfl" = %s
        LIMIT 1;
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, (pinfl,))
        result = cur.fetchone()
        conn.close()

        if not result or not result.get("credit_data"):
            return None

        data = result["credit_data"]
        if isinstance(data, str):
            data = json.loads(data)

        if isinstance(data, list):
            return {"contracts": data}
        elif isinstance(data, dict):
            return data
        else:
            print(f"[Warning] Unexpected credit format for PINFL {pinfl}")
            return None
    except Exception as e:
        print(f"[Error] Failed to load credit for {pinfl}: {e}")
        return None

if __name__ == "__main__":
    # 🔍 Test shared PINFLs
    shared_pinfls = get_shared_pinfls()
    print(f"✅ Found {len(shared_pinfls)} PINFLs with both credit & salary data.")

    # 👁️ Test loading one
    if shared_pinfls:
        test_pinfl = shared_pinfls[0]
        salary = load_salary_json(test_pinfl)
        credit = load_credit_json(test_pinfl)

        print(f"\n📦 Salary for {test_pinfl}:\n{json.dumps(salary, indent=2)}")
        print(f"\n📄 Credit for {test_pinfl}:\n{json.dumps(credit, indent=2)}")
