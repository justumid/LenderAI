import logging
from typing import List, Optional, Any, Dict, Union
from dateutil.parser import parse
import pandas as pd

logger = logging.getLogger(__name__)

# === Constants ===
DEFAULT_INFERRED_INCOME_TAX_RATE = 0.12
DEBT_DELTA_SMOOTHING_THRESHOLD_FACTOR = 0.5
ANOMALY_FLAG_DEBT_DELTA_FACTOR = 0.75
ANOMALY_FLAG_OVERDUE_PERCENT_THRESHOLD = 0.5
ANOMALY_FLAG_PTI_THRESHOLD = 0.7
PTI_HIGH_VALUE_INDICATOR = 999.0

# === Utilities ===
def safe_float(val: Any, default_val: float = 0.0) -> float:
    try: return float(str(val).replace(',', ''))
    except: return default_val

def safe_int(val: Any, default_val: int = 0) -> int:
    try: return int(str(val).replace(',', ''))
    except:
        try: return int(float(str(val).replace(',', '')))
        except: return default_val

def format_month_to_yyyy_mm(date_str: Optional[str]) -> Optional[str]:
    if not date_str: return None
    try: return parse(str(date_str).strip()).strftime("%Y-%m")
    except Exception: return None

def _create_monthly_timeline(max_length: int, default_value: Union[float, int] = 0.0, current_date_for_timeline: Optional[pd.Timestamp] = None) -> pd.Series:
    end_date = current_date_for_timeline or pd.Timestamp.now()
    date_index = pd.date_range(end=end_date.normalize(), periods=max_length, freq='MS')
    dtype = float if isinstance(default_value, float) else int
    return pd.Series(default_value, index=date_index.strftime('%Y-%m'), dtype=dtype)

# === SequenceExtractor ===
class SequenceExtractor:
    def __init__(self, salary_data: Optional[Dict[str, Any]], credit_data: Optional[Dict[str, Any]], default_max_sequence_length: int = 24, current_date_for_timeline: Optional[str] = None, config: Optional[Dict[str, float]] = None):
        self.salary_data = salary_data or {}
        self.credit_data = credit_data or {}
        self.default_max_sequence_length = default_max_sequence_length

        self.current_ts = pd.Timestamp(current_date_for_timeline) if current_date_for_timeline else pd.Timestamp.now()

        cfg = config or {}
        self.inferred_tax_rate = cfg.get('INFERRED_INCOME_TAX_RATE', DEFAULT_INFERRED_INCOME_TAX_RATE)
        self.debt_delta_smooth_factor = cfg.get('DEBT_DELTA_SMOOTHING_FACTOR', DEBT_DELTA_SMOOTHING_THRESHOLD_FACTOR)
        self.anomaly_debt_delta_factor = cfg.get('ANOMALY_DEBT_DELTA_FACTOR', ANOMALY_FLAG_DEBT_DELTA_FACTOR)
        self.anomaly_overdue_pct_thresh = cfg.get('ANOMALY_OVERDUE_PCT_THRESHOLD', ANOMALY_FLAG_OVERDUE_PERCENT_THRESHOLD)
        self.anomaly_pti_thresh = cfg.get('ANOMALY_PTI_THRESHOLD', ANOMALY_FLAG_PTI_THRESHOLD)
        self.pti_high_fill_value = cfg.get('PTI_HIGH_VALUE', PTI_HIGH_VALUE_INDICATOR)

    def _get_max_len(self, max_length: Optional[int]) -> int:
        return max_length or self.default_max_sequence_length

    def get_salary_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        timeline = _create_monthly_timeline(max_len, 0.0, self.current_ts)
        for r in self.salary_data.get("salary_records", []):
            month_key = format_month_to_yyyy_mm(r.get("payment_date"))
            if month_key in timeline.index:
                income = safe_float(r.get("income_amount", 0.0))
                if income == 0.0 and self.inferred_tax_rate > 0:
                    tax = safe_float(r.get("tax_sum", 0.0))
                    income = tax / self.inferred_tax_rate if tax > 0 else income
                timeline[month_key] += income
        return timeline.tolist()

    def get_salary_tax_sequence(self, max_length: Optional[int] = None) -> List[Dict[str, float]]:
        max_len = self._get_max_len(max_length)
        timeline_index = _create_monthly_timeline(max_len, current_date_for_timeline=self.current_ts).index
        monthly_data = {key: {"income": 0.0, "tax": 0.0, "tax_free": 0.0} for key in timeline_index}
        for r in self.salary_data.get("salary_records", []):
            month_key = format_month_to_yyyy_mm(r.get("payment_date"))
            if month_key in monthly_data:
                income = safe_float(r.get("income_amount", 0.0))
                tax = safe_float(r.get("tax_sum", 0.0))
                if income == 0.0 and tax > 0 and self.inferred_tax_rate > 0:
                    income = tax / self.inferred_tax_rate
                tax_free = max(income - tax, 0.0)
                monthly_data[month_key]["income"] += income
                monthly_data[month_key]["tax"] += tax
                monthly_data[month_key]["tax_free"] += tax_free
        return [monthly_data[key] for key in timeline_index]

    def get_debt_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        timeline = _create_monthly_timeline(max_len, 0.0, self.current_ts)
        for contract in self.credit_data.get("contracts", []):
            for balance in contract.get("balances", []):
                month_key = format_month_to_yyyy_mm(balance.get("month"))
                if month_key in timeline.index:
                    timeline[month_key] += safe_float(balance.get("end_sum", 0.0))
        return timeline.tolist()

    def get_debt_delta_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        debt_seq_full = list(reversed(self.get_debt_sequence(max_len + 1)))
        deltas = []
        for i in range(max_len):
            if i + 1 < len(debt_seq_full):
                prev_debt, current_debt = debt_seq_full[i], debt_seq_full[i+1]
                delta = current_debt - prev_debt
                if abs(delta) > self.debt_delta_smooth_factor * max(abs(prev_debt), abs(current_debt), 1.0):
                    delta = 0.0
                deltas.append(delta)
            else:
                deltas.append(0.0)
        return list(reversed(deltas))

    def get_actual_repayment_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        timeline = _create_monthly_timeline(max_len, 0.0, self.current_ts)
        for contract in self.credit_data.get("contracts", []):
            for balance in contract.get("balances", []):
                month_key = format_month_to_yyyy_mm(balance.get("month"))
                if month_key in timeline.index:
                    begin_sum = safe_float(balance.get("begin_sum", 0.0))
                    end_sum = safe_float(balance.get("end_sum", 0.0))
                    repayment = begin_sum - end_sum
                    if repayment > 0:
                        timeline[month_key] += repayment
        return timeline.tolist()

    def get_overdue_days_sequence(self, max_length: Optional[int] = None) -> List[int]:
        max_len = self._get_max_len(max_length)
        timeline = _create_monthly_timeline(max_len, 0, self.current_ts)
        for contract in self.credit_data.get("contracts", []):
            for overdue_info in contract.get("overdue_procents", []):
                month_key = format_month_to_yyyy_mm(overdue_info.get("overdue_date"))
                if month_key in timeline.index:
                    days = safe_int(overdue_info.get("overdue_principal_days", 0))
                    timeline[month_key] = max(timeline[month_key], days)
        return timeline.tolist()

    def get_overdue_flag_sequence(self, max_length: Optional[int] = None) -> List[int]:
        return [1 if days > 0 else 0 for days in self.get_overdue_days_sequence(max_length)]

    def get_overdue_sum_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        timeline = _create_monthly_timeline(max_len, 0.0, self.current_ts)
        for contract in self.credit_data.get("contracts", []):
            for overdue_info in contract.get("overdue_procents", []):
                month_key = format_month_to_yyyy_mm(overdue_info.get("overdue_date"))
                if month_key in timeline.index:
                    timeline[month_key] += safe_float(overdue_info.get("overdue_principal_sum", 0.0))
        return timeline.tolist()

    def get_overdue_percent_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        timeline = _create_monthly_timeline(max_len, 0.0, self.current_ts)
        for contract in self.credit_data.get("contracts", []):
            contract_amount = safe_float(contract.get("amount", 0.0))
            if contract_amount <= 0:
                continue
            for overdue_info in contract.get("overdue_procents", []):
                month_key = format_month_to_yyyy_mm(overdue_info.get("overdue_date"))
                if month_key in timeline.index:
                    overdue_sum = safe_float(overdue_info.get("overdue_principal_sum", 0.0))
                    percent = (overdue_sum / contract_amount) * 100.0
                    timeline[month_key] = max(timeline[month_key], percent)
        return timeline.tolist()

    def get_payment_to_income_sequence(self, max_length: Optional[int] = None) -> List[float]:
        max_len = self._get_max_len(max_length)
        timeline_index = _create_monthly_timeline(max_len, current_date_for_timeline=self.current_ts).index
        salary_series = pd.Series(self.get_salary_sequence(max_len), index=timeline_index)
        repayment_series = pd.Series(self.get_actual_repayment_sequence(max_len), index=timeline_index)
        pti_values = []
        for month_key in timeline_index:
            salary = salary_series.get(month_key, 0.0)
            repayment = repayment_series.get(month_key, 0.0)
            if salary >= 1.0:
                pti = repayment / salary
            elif repayment > 0:
                pti = self.pti_high_fill_value
            else:
                pti = 0.0
            pti_values.append(min(pti, self.pti_high_fill_value))
        return pti_values

    def get_fraud_sequence(self, max_length: Optional[int] = None) -> List[List[float]]:
        max_len = self._get_max_len(max_length)
        debt_s = self.get_debt_sequence(max_len)
        overdue_f_s = self.get_overdue_flag_sequence(max_len)
        salary_s = self.get_salary_sequence(max_len)
        overdue_sum_s = self.get_overdue_sum_sequence(max_len)
        overdue_pct_s = self.get_overdue_percent_sequence(max_len)
        repayment_s = self.get_actual_repayment_sequence(max_len)
        pti_s = self.get_payment_to_income_sequence(max_len)
        debt_delta_s = self.get_debt_delta_sequence(max_len)

        fraud_seq = []
        for i in range(max_len):
            is_anomalous = (
                (abs(debt_delta_s[i]) > self.anomaly_debt_delta_factor * max(abs(debt_s[i]), 1.0)) or
                (overdue_pct_s[i] > self.anomaly_overdue_pct_thresh) or
                (pti_s[i] > self.anomaly_pti_thresh and salary_s[i] > 0) or
                (overdue_f_s[i] == 1 and repayment_s[i] == 0.0)
            )
            fraud_seq.append([
                debt_s[i], float(overdue_f_s[i]), salary_s[i],
                overdue_sum_s[i], overdue_pct_s[i], repayment_s[i],
                pti_s[i], debt_delta_s[i], float(is_anomalous)
            ])
        return fraud_seq

    def extract(self, max_length: Optional[int] = None) -> Dict[str, List[Any]]:
        max_len = self._get_max_len(max_length)
        return {
            "salary": self.get_salary_sequence(max_len),
            "salary_tax": self.get_salary_tax_sequence(max_len),
            "debt": self.get_debt_sequence(max_len),
            "debt_delta": self.get_debt_delta_sequence(max_len),
            "overdue_days": self.get_overdue_days_sequence(max_len),
            "overdue_flags": self.get_overdue_flag_sequence(max_len),
            "overdue_sum": self.get_overdue_sum_sequence(max_len),
            "overdue_percent": self.get_overdue_percent_sequence(max_len),
            "repayment": self.get_actual_repayment_sequence(max_len),
            "payment_to_income": self.get_payment_to_income_sequence(max_len),
            "fraud_sequence": self.get_fraud_sequence(max_len)
        }
