import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LimitDecisionService:
    def __init__(self):
        logger.info("✅ LimitDecisionService initialized.")

    def decide_segmented(
        self,
        features: Dict[str, Any],
        pd_score: float,
        ead: float,
        lgd: float,
        segment: str,
        static_score: float,
        fraud_score: float
    ) -> Tuple[float, Dict[str, Any]]:
        explanation = {}

        # === Base Limit by PD Score ===
        if pd_score < 0.05:
            base_limit = 100_000_000
        elif pd_score < 0.10:
            base_limit = 70_000_000
        elif pd_score < 0.20:
            base_limit = 50_000_000
        elif pd_score < 0.30:
            base_limit = 30_000_000
        else:
            base_limit = 15_000_000  # NO REJECTION, just heavy penalty

        explanation["base_limit_by_pd"] = base_limit
        explanation["pd_score"] = pd_score

        # === Segment Scaling ===
        segment_boost = {"low": 1.2, "medium": 1.0, "high": 0.9}.get(segment, 1.0)

        # === Salary Scaling ===
        salary = features.get("salary_mean", 0.0)
        salary_boost = 1.2 if salary >= 20_000_000 else 1.1 if salary >= 10_000_000 else 1.0 if salary >= 5_000_000 else 0.9

        # === PTI Penalty ===
        monthly_payment = features.get("overview_actual_avg_payment", 0.0)
        pti_ratio = monthly_payment / salary if salary > 0 else 0.0
        pti_penalty = 0.9 if pti_ratio > 0.5 else 1.0
        if pti_ratio > 1.0:
            pti_penalty = 0.8

        # === Fraud & Overdue Penalties ===
        overdue_ratio = features.get("open_overdue_ratio", 0.0)
        fraud_penalty = 1.0 - min(fraud_score * 0.3, 0.3)
        overdue_penalty = 1.0 - min(overdue_ratio * 1.5, 0.3)

        # === Credit Request Penalty ===
        credit_request_qty = features.get("overview_credit_request_qty", 0)
        request_penalty = 0.95 if credit_request_qty >= 50 else 1.0

        # === DTI Penalty ===
        total_debt = features.get("total_open_debt", 0.0)
        dti_ratio = total_debt / salary if salary > 0 else 1.0
        dti_penalty = 0.95 if dti_ratio > 3 else 1.0
        if dti_ratio > 5:
            dti_penalty = 0.85

        # === Salary Growth Bonus ===
        salary_growth = features.get("salary_growth", 0.0)
        growth_bonus = 1.05 if salary_growth > 0 else 1.0

        # === Static Score Influence ===
        static_score_boost = 1.1 if static_score >= 0.8 else 0.9 if static_score <= 0.5 else 1.0

        # === Final Limit Computation ===
        limit = base_limit
        limit *= segment_boost
        limit *= salary_boost
        limit *= pti_penalty
        limit *= fraud_penalty
        limit *= overdue_penalty
        limit *= request_penalty
        limit *= dti_penalty
        limit *= growth_bonus
        limit *= static_score_boost

        # === Ensure Minimum Limit ===
        limit = max(limit, 5_000_000)

        # === Cap by requested amount ===
        requested_amount = features.get("requested_amount", 100_000_000)
        cap_limit = max(ead, requested_amount) * 1.2
        limit = min(limit, cap_limit)
        limit = round(limit, -4)

        # === Explanation Output ===
        explanation.update({
            "segment": segment,
            "segment_boost": segment_boost,
            "salary_mean": salary,
            "salary_boost": salary_boost,
            "monthly_payment": monthly_payment,
            "pti_ratio": pti_ratio,
            "pti_penalty": pti_penalty,
            "fraud_score": fraud_score,
            "fraud_penalty": fraud_penalty,
            "overdue_ratio": overdue_ratio,
            "overdue_penalty": overdue_penalty,
            "credit_request_qty": credit_request_qty,
            "request_penalty": request_penalty,
            "dti_ratio": dti_ratio,
            "dti_penalty": dti_penalty,
            "salary_growth": salary_growth,
            "growth_bonus": growth_bonus,
            "static_score": static_score,
            "static_score_boost": static_score_boost,
            "requested_amount": requested_amount,
            "cap_limit": cap_limit,
            "final_limit": limit,
            "final_reason": "Adjusted by PD, fraud, overdue, affordability & static score"
        })

        return limit, explanation
