import datetime
import logging
from statistics import mean, stdev
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FeatureExtractor:
    def __init__(self, salary_data: Dict[str, Any], credit_data: Dict[str, Any]):
        self.salary = salary_data or {}
        self.credit = credit_data or {}

        self.contracts = self.credit.get("contracts", [])
        self.open_contracts = [c for c in self.contracts if str(c.get("contract_status")) == "1"]
        self.contingent_liabs = self.credit.get("contingent_liabilities", [])

    def _safe_float(self, val: Any) -> float:
        try:
            return float(val)
        except Exception:
            return 0.0

    def _safe_int(self, val: Any) -> int:
        try:
            return int(val)
        except Exception:
            return 0

    def extract_salary_features(self) -> Dict[str, float]:
        records = self.salary.get("salary_records", [])
        if not records:
            return {k: 0.0 for k in [
                "salary_mean", "salary_std", "salary_growth", "salary_min", "salary_max",
                "salary_last_6mo_avg", "salary_tenure_months", "salary_record_count", "has_salary_data"
            ]}

        income_vals, valid_records, dates = [], [], []
        for r in records:
            income = self._safe_float(r.get("income_amount"))
            tax = self._safe_float(r.get("tax_sum"))
            if income == 0.0 and tax > 0:
                income = tax / 0.12
            if income > 0 and r.get("payment_date"):
                income_vals.append(income)
                valid_records.append((r["payment_date"], income))
                dates.append(r["payment_date"])

        salary_mean_val = mean(income_vals) if income_vals else 0.0
        salary_std_val = stdev(income_vals) if len(income_vals) > 1 else 0.0
        salary_min_val = min(income_vals) if income_vals else 0.0
        salary_max_val = max(income_vals) if income_vals else 0.0

        sorted_income = sorted(valid_records, key=lambda x: x[0])
        salary_growth = 0.0
        if len(sorted_income) >= 2:
            first, last = sorted_income[0][1], sorted_income[-1][1]
            if first > 500_000 and last > 500_000:
                salary_growth = (last - first) / first

        last_6mo_vals = [v for _, v in sorted_income[-6:]]
        salary_last_6mo_avg = mean(last_6mo_vals) if len(last_6mo_vals) >= 2 else salary_mean_val

        tenure_months = 0
        if len(dates) >= 2:
            try:
                start = datetime.datetime.strptime(dates[0], "%Y-%m-%d")
                end = datetime.datetime.strptime(dates[-1], "%Y-%m-%d")
                tenure_months = (end.year - start.year) * 12 + (end.month - start.month)
            except:
                pass
        if tenure_months == 0 and len(dates) >= 1:
            tenure_months = len(dates)

        return {
            "salary_mean": salary_mean_val,
            "salary_std": salary_std_val,
            "salary_growth": salary_growth,
            "salary_min": salary_min_val,
            "salary_max": salary_max_val,
            "salary_last_6mo_avg": salary_last_6mo_avg,
            "salary_tenure_months": tenure_months,
            "salary_record_count": len(records),
            "has_salary_data": float(bool(records)),
        }

    def extract_credit_features(self) -> Dict[str, float]:
        total_contingent_debt = sum(self._safe_float(c.get("total_debt_sum")) for c in self.contingent_liabs)
        total_contingent_overdue = sum(self._safe_float(c.get("overdue_debt_sum")) for c in self.contingent_liabs)
        contingent_ratio = total_contingent_overdue / total_contingent_debt if total_contingent_debt > 0 else 0.0

        total_open_debt, total_open_overdue, total_open_payment = 0.0, 0.0, 0.0
        open_debts = []

        for contract in self.open_contracts:
            for balance in contract.get("balances", []):
                end_sum = self._safe_float(balance.get("end_sum"))
                total_open_debt += end_sum
                open_debts.append(end_sum)
                total_open_payment += self._safe_float(balance.get("monthly_average_payment", 0.0))

            for overdue in contract.get("overdue_procents", []):
                total_open_overdue += self._safe_float(overdue.get("overdue_principal_sum"))

            if total_open_overdue == 0.0:
                total_open_overdue += self._safe_float(contract.get("overdue_debt_sum"))

        open_overdue_ratio = total_open_overdue / total_open_debt if total_open_debt > 0 else 0.0
        avg_open_payment = total_open_payment / len(self.open_contracts) if self.open_contracts else 0.0

        return {
            "num_open_contracts": len(self.open_contracts),
            "num_total_contracts": len(self.contracts),
            "num_contingent_liabilities": len(self.contingent_liabs),
            "total_contingent_debt": total_contingent_debt,
            "total_contingent_overdue": total_contingent_overdue,
            "contingent_overdue_ratio": contingent_ratio,
            "total_open_debt": total_open_debt,
            "total_open_overdue": total_open_overdue,
            "open_overdue_ratio": open_overdue_ratio,
            "avg_open_monthly_payment": avg_open_payment,
            "max_open_contract_debt": max(open_debts) if open_debts else 0.0,
            "min_open_contract_debt": min(open_debts) if open_debts else 0.0,
            "has_open_contracts": float(bool(self.open_contracts)),
        }

    def extract_overview_features(self) -> Dict[str, float]:
        overview = self.credit.get("overview", {})
        return {
            "overview_contracts_qty": self._safe_int(overview.get("contracts_qty")),
            "overview_credit_request_qty": self._safe_int(overview.get("credit_request_qty")),
            "overview_avg_monthly_payment": self._safe_float(overview.get("average_monthly_payment")),
            "overview_actual_avg_payment": self._safe_float(overview.get("actual_average_monthly_payment")),
            "overview_overdue_principal_qty": self._safe_int(overview.get("overdue_principal_qty")),
            "has_overview": float(bool(overview)),
        }

    def extract_client_features(self) -> Dict[str, float]:
        client = self.credit.get("client", {})
        age = self._safe_int(client.get("age"))
        if not age and client.get("birth_date"):
            try:
                dob = datetime.datetime.strptime(client["birth_date"], "%Y-%m-%d")
                age = (datetime.datetime.today() - dob).days // 365
            except:
                age = 0
        phone_count = len(client.get("phones", [])) if isinstance(client.get("phones"), list) else 0
        return {
            "client_age": age,
            "client_phone_count": phone_count,
            "has_client": float(bool(client)),
        }

    def extract_credit_request_features(self) -> Dict[str, float]:
        requests = self.credit.get("credit_requests", [])
        if isinstance(requests, dict):
            requests = requests.get("credit_request", [])

        parsed_dates = []
        for r in requests:
            dt = r.get("demand_date_time")
            if dt:
                try:
                    parsed_dates.append(datetime.datetime.strptime(dt.split()[0], "%Y-%m-%d"))
                except:
                    pass

        last_days_ago = (datetime.datetime.today() - max(parsed_dates)).days if parsed_dates else 9999

        return {
            "num_credit_requests": len(requests),
            "days_since_last_credit_request": last_days_ago,
            "has_credit_requests": float(bool(requests)),
        }

    def extract_scoring_features(self) -> Dict[str, Any]:
        scoring = self.credit.get("scorring", {})
        return {
            "scoring_grade": self._safe_int(scoring.get("scoring_grade")),
            "scoring_version": scoring.get("scoring_version") or "",
            "scoring_class": scoring.get("scoring_class") or "",
            "scoring_level": scoring.get("scoring_level") or "",
            "anomaly_score": self._safe_float(scoring.get("anomaly_score")),
            "pd_label": self._safe_float(scoring.get("pd_label")),
        }

    def extract_all_features(self) -> Dict[str, Any]:
        salary = self.extract_salary_features()
        credit = self.extract_credit_features()
        overview = self.extract_overview_features()
        client = self.extract_client_features()
        credit_request = self.extract_credit_request_features()
        scoring = self.extract_scoring_features()

        features = {**salary, **credit, **overview, **client, **credit_request, **scoring}

        features["katm_score"] = features.get("scoring_grade", -1)
        features["katm_score_normalized"] = min(max((features["katm_score"] - 200) / 300, 0.0), 1.0)

        salary_base = features["salary_last_6mo_avg"] if features["salary_last_6mo_avg"] > 0 else features["salary_mean"]
        features["pti_ratio"] = features["overview_actual_avg_payment"] / salary_base if salary_base else 0.0
        features["dti_ratio"] = features["total_open_debt"] / salary_base if salary_base else 1.0

        features["overdue_risk_flag"] = 1.0 if features["open_overdue_ratio"] >= 0.2 else 0.0
        features["debt_utilization"] = features["total_open_debt"] + features["total_contingent_debt"]

        weighted_income = (features["salary_mean"] * 0.6) + (features["salary_last_6mo_avg"] * 0.4)
        safe_payment = weighted_income * 0.3

        if features["dti_ratio"] > 0.5:
            safe_payment *= 0.8
        if features["anomaly_score"] >= 0.5:
            safe_payment *= 0.7
        safe_payment *= (1 - features["pd_label"])

        if features["scoring_class"] in ["trusted", "vip"]:
            safe_payment *= 1.1
        elif features["scoring_class"] in ["risky", "watchlist"]:
            safe_payment *= 0.8

        features["safe_payment_capability"] = safe_payment

        features["__source_flags__"] = {
            "has_client": bool(client),
            "has_credit_requests": bool(credit_request),
            "has_overview": bool(overview),
            "has_open_contracts": bool(self.open_contracts),
        }

        return features
