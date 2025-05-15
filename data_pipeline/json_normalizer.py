import json
import logging
from typing import Optional, Any, Dict, List, Union

logger = logging.getLogger("json_normalizer")
logging.basicConfig(level=logging.INFO)


def safe_float(val: Any) -> float:
    try:
        return float(str(val).replace(",", "").strip())
    except (ValueError, TypeError, AttributeError):
        return 0.0


def safe_int(val: Any) -> int:
    try:
        return int(float(str(val).replace(",", "").strip()))
    except (ValueError, TypeError, AttributeError):
        return 0


def _ensure_list(val: Any) -> List:
    if isinstance(val, list):
        return val
    elif isinstance(val, dict):
        return [val]
    return []

def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default



def normalize_salary_json(raw: Optional[Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"salary_records": []}

    try:
        # Handle stringified JSON
        if isinstance(raw, str):
            raw = json.loads(raw)

        # Wrap into list if necessary
        if isinstance(raw, dict):
            records = (
                raw.get("salary_data") or
                raw.get("records") or
                raw.get("salary_records") or []
            )
        elif isinstance(raw, list):
            records = raw
        else:
            records = []

        records = _ensure_list(records)

        for rec in records:
            year = rec.get("year")
            period = rec.get("period")

            # Try to construct payment date
            payment_date = None
            try:
                if year and period:
                    payment_date = f"{int(year)}-{int(period):02d}-01"
            except Exception:
                logger.warning(f"[normalize_salary_json] Invalid date from year={year}, period={period}")

            # Extract income and tax
            income_amount = safe_float(
                rec.get("salary") or
                rec.get("income") or
                rec.get("amount") or
                rec.get("income_amount")
            )
            tax_sum = safe_float(rec.get("salaryTaxSum") or rec.get("tax_sum"))

            # Recover income from tax if missing
            if income_amount == 0.0 and tax_sum > 0.0:
                income_amount = tax_sum / 0.12

            tax_free = income_amount - tax_sum if income_amount > tax_sum else 0.0

            # Add record regardless of payment_date presence
            out["salary_records"].append({
                "income_amount": income_amount,
                "tax_sum": tax_sum,
                "tax_free": tax_free,
                "payment_date": payment_date
            })

        if not out["salary_records"]:
            logger.warning("[normalize_salary_json] No valid salary records parsed.")

    except Exception as e:
        logger.warning(f"[normalize_salary_json] Failed to normalize: {e}")

    return out

def normalize_credit_json(raw: Optional[Any]) -> Dict[str, Any]:
    out = {
        "contracts": [],
        "open_contracts": [],
        "contingent_liabilities": [],
        "overview": {},
        "credit_requests": [],
        "client": {},
        "scorring": {},
        "meta": {}
    }

    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
        if isinstance(raw, list):
            raw = raw[0] if raw else {}

        report = raw.get("report", {})
        out["meta"]["pinfl"] = raw.get("test") or report.get("client", {}).get("pinfl")

        # === CONTRACTS ===
        contracts_raw = report.get("contracts", {}).get("contract", [])
        if isinstance(contracts_raw, str):
            try:
                contracts_raw = json.loads(contracts_raw)
            except Exception:
                contracts_raw = []
        contracts = _ensure_list(contracts_raw)

        parsed, skipped = 0, 0

        for contract in contracts:
            try:
                if isinstance(contract, str):
                    contract = json.loads(contract)
                if not isinstance(contract, dict):
                    skipped += 1
                    continue

                contract_obj = {
                    "contract_id": contract.get("contract_id", ""),
                    "contract_status": contract.get("contract_status", ""),
                    "contract_date": contract.get("contract_date"),
                    "contract_end_date": contract.get("contract_end_date"),
                    "org_name": contract.get("org_name"),
                    "credit_type": contract.get("credit_type"),
                    "amount": safe_float(contract.get("amount")) / 100,
                    "total_debt_sum": safe_float(contract.get("total_debt_sum")) / 100,
                    "immediate_principal_sum": safe_float(contract.get("immediate_principal_sum")) / 100,
                    "monthly_average_payment": 0.0,
                    "overdue_debt_sum": safe_float(contract.get("overdue_debt_sum")) / 100,
                    "max_uninter_overdue_percent": safe_float(contract.get("max_uninter_overdue_percent")),
                    "balances": [],
                    "overdue_procents": [],
                    "actual_repayments": [],
                    "forecasted_payments": []
                }

                # === BALANCES ===
                balances_raw = contract.get("balances")
                balance_list = _ensure_list(balances_raw.get("balance", [])) if isinstance(balances_raw, dict) else balances_raw if isinstance(balances_raw, list) else []
                repayment_deltas = []
                for b in balance_list:
                    begin = safe_float(b.get("begin_sum")) / 100
                    end = safe_float(b.get("end_sum")) / 100
                    delta = max(0.0, begin - end)
                    if delta > 0:
                        repayment_deltas.append(delta)
                    contract_obj["balances"].append({
                        "month": b.get("month"),
                        "begin_sum": begin,
                        "end_sum": end
                    })

                # === OVERDUE PROCENTS ===
                overdue_raw = contract.get("overdue_procents")
                overdue_list = _ensure_list(overdue_raw.get("overdue_procent", [])) if isinstance(overdue_raw, dict) else overdue_raw if isinstance(overdue_raw, list) else []
                for o in overdue_list:
                    contract_obj["overdue_procents"].append({
                        "overdue_principal_sum": safe_float(o.get("overdue_principal_sum")) / 100,
                        "overdue_date": o.get("overdue_date"),
                        "overdue_principal_days": safe_int(o.get("overdue_principal_days")),
                        "overdue_principal_change": o.get("overdue_principal_change")
                    })

                # === ACTUAL REPAYMENTS ===
                actual_raw = contract.get("actual_schedule")
                actual_list = _ensure_list(actual_raw.get("actual_repayment", [])) if isinstance(actual_raw, dict) else []
                for r in actual_list:
                    contract_obj["actual_repayments"].append({
                        "repayment_date": r.get("repayment_date"),
                        "principal_sum": safe_float(r.get("principal_sum")) / 100,
                        "percent_sum": safe_float(r.get("percent_sum")) / 100,
                        "remaining_principal_sum": safe_float(r.get("remaining_principal_sum")) / 100,
                    })

                # === FORECASTED REPAYMENTS ===
                forecasted_raw = contract.get("forecasted_schedule")
                forecasted_list = _ensure_list(forecasted_raw.get("forecasted_payment", [])) if isinstance(forecasted_raw, dict) else []
                for f in forecasted_list:
                    contract_obj["forecasted_payments"].append({
                        "period": f.get("forecasted_payment_period"),
                        "principal_sum": safe_float(f.get("principal_sum")) / 100,
                        "percent_sum": safe_float(f.get("percent_sum")) / 100,
                    })

                # === MONTHLY PAYMENT CALCULATION ===
                if len(repayment_deltas) >= 2:
                    contract_obj["monthly_average_payment"] = sum(repayment_deltas) / len(repayment_deltas)
                elif len(contract_obj["actual_repayments"]) >= 3:
                    reps = sorted(contract_obj["actual_repayments"], key=lambda x: x.get("repayment_date") or "")[-3:]
                    contract_obj["monthly_average_payment"] = sum(
                        safe_float(r.get("principal_sum")) + safe_float(r.get("percent_sum"))
                        for r in reps
                    ) / 3
                else:
                    contract_obj["monthly_average_payment"] = safe_float(contract.get("monthly_average_payment")) / 100

                out["contracts"].append(contract_obj)
                if str(contract_obj["contract_status"]).strip().lower() in {"1", "open"}:
                    out["open_contracts"].append(contract_obj)
                parsed += 1

            except Exception as e:
                skipped += 1
                logger.warning(f"[Contract Normalize] Skipped contract: {e}")

        logger.info(f"[normalize_credit_json] Parsed {parsed}, skipped {skipped} contract(s).")

        # === CONTINGENT LIABILITIES ===
        liabilities = _ensure_list(report.get("contingent_liabilities", {}).get("contingent_liability", []))
        for l in liabilities:
            l["overdue_debt_sum"] = safe_float(l.get("overdue_debt_sum")) / 100
            l["max_uninter_overdue_percent"] = safe_float(l.get("max_uninter_overdue_percent"))
        out["contingent_liabilities"] = liabilities

        # === OVERVIEW ===
        overview = report.get("overview", {}) or {}
        for key, value in overview.items():
            try:
                fval = float(value)
                overview[key] = fval / 100 if "sum" in key or "payment" in key else fval
            except Exception:
                continue
        out["overview"] = overview

        # === CREDIT REQUESTS ===
        out["credit_requests"] = _ensure_list(report.get("credit_requests", {}).get("credit_request", []))

        # === CLIENT INFO ===
        client = raw.get("client", report.get("client", {})) or {}
        if isinstance(client.get("phones"), dict):
            client["phones"] = _ensure_list(client["phones"].get("phone", []))
        elif not isinstance(client.get("phones"), list):
            client["phones"] = []
        out["client"] = client

        # === SCORRING ===
        scoring_raw = raw.get("scorring", report.get("scorring", {})) or {}
        out["scorring"] = {
            "scoring_grade": safe_int(scoring_raw.get("scoring_grade", -1)),
            "scoring_version": scoring_raw.get("scoring_version", "unknown"),
            "scoring_class": scoring_raw.get("scoring_class", "unknown"),
            "scoring_level": scoring_raw.get("scoring_level", "unknown")
        }

    except Exception as e:
        logger.warning(f"[normalize_credit_json] Failed to normalize: {e}")

    return out
# === Test driver ===
if __name__ == "__main__":
    import pprint
    from data_ingestion import load_credit_json, load_salary_json
    from json_normalizer import normalize_credit_json, normalize_salary_json

    test_pinfl = "30303901621130"

    # === Load raw data
    raw_credit = load_credit_json(test_pinfl)
    raw_salary = load_salary_json(test_pinfl)

    print(f"[DEBUG] Raw salary JSON for {test_pinfl}: {type(raw_salary)}")
    print(f"[DEBUG] Raw credit JSON (1st level) for {test_pinfl}: {type(raw_credit)}")

    # === Normalize
    credit = normalize_credit_json(raw_credit)
    salary = normalize_salary_json(raw_salary)

    # === PINFL
    print("\n🆔 PINFL:", credit.get("meta", {}).get("pinfl", "N/A"))

    # === Overview
    print("\n📦 Overview:")
    pprint.pprint(credit.get("overview", {}))

    # === Client info
    print("\n👤 Client Info:")
    pprint.pprint(credit.get("client", {}))

    # === Contracts
    contracts = credit.get("contracts", [])
    open_contracts = credit.get("open_contracts", [])
    print(f"\n📊 Total Contracts: {len(contracts)}")
    print(f"📊 Open Contracts: {len(open_contracts)}")

    if contracts:
        print("\n🔍 First Contract (any):")
        pprint.pprint(contracts[0])
    else:
        print("⚠️ No contracts found.")

    # === Contingent liabilities & requests
    print("\n📄 Contingent Liabilities:", len(credit.get("contingent_liabilities", [])))
    print("📄 Credit Requests:", len(credit.get("credit_requests", [])))

    # === Salary records
    print("\n💼 Salary Records:")
    records = salary.get("salary_records", [])
    if records:
        pprint.pprint(records[:3])
        print(f"... total {len(records)} salary records")
    else:
        print("⚠️ No salary records found.")
