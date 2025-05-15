from typing import Dict, Union


class ReviewerSimulator:
    """
    Simulates human-like loan review decisions.
    Factors:
    - Final ML score
    - Fraud score
    - Segment bias
    - Loan limit feasibility
    - Override behavior
    """

    def __init__(
        self,
        low_threshold: float = 0.45,
        high_threshold: float = 0.7,
        fraud_threshold: float = 0.4,
        min_loan_limit: float = 3_000_000
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.fraud_threshold = fraud_threshold
        self.min_loan_limit = min_loan_limit

        # Segment-based reviewer tendencies (lower score threshold)
        self.segment_bonuses = {
            "high_income_reliable": -0.05,
            "low_income": +0.05,
            "overdue_unstable": +0.08,
            "bureau_risk": +0.1,
        }

    def review(
        self,
        final_score: float,
        fraud_score: float,
        segment: str,
        loan_limit: float
    ) -> Dict[str, Union[str, float, bool]]:
        """
        Simulate a reviewer decision based on risk and behavior context.
        Returns decision + reason + override flag + effective threshold used.
        """

        explanation = ""
        override = False
        decision = "refer"
        effective_threshold = self.low_threshold

        # === Fraud hard block
        if fraud_score >= self.fraud_threshold:
            return {
                "decision": "reject",
                "reason": f"Fraud anomaly score too high ({fraud_score:.2f})",
                "manual_override": False,
                "effective_threshold": -1
            }

        # === Adjust threshold per segment
        bias = self.segment_bonuses.get(segment, 0.0)
        effective_threshold += bias

        # === Manual override: strong segment + medium score
        if final_score >= self.high_threshold:
            if segment == "high_income_reliable" and fraud_score < 0.2:
                explanation = f"Override: strong segment ({segment}) despite high score"
                override = True
                decision = "approve"
            else:
                return {
                    "decision": "reject",
                    "reason": f"Score too high to approve ({final_score:.2f})",
                    "manual_override": False,
                    "effective_threshold": effective_threshold
                }

        elif final_score >= effective_threshold:
            if loan_limit >= self.min_loan_limit:
                decision = "approve"
                explanation = "Sufficient limit and acceptable score"
            else:
                decision = "refer"
                explanation = "Score ok but limit too small"

        elif final_score < effective_threshold:
            if segment == "low_income" and fraud_score < 0.15:
                decision = "refer"
                explanation = "Low-income segment, borderline risk"
            else:
                decision = "reject"
                explanation = "Score and limit both too weak"

        return {
            "decision": decision,
            "reason": explanation or "Threshold-based decision",
            "manual_override": override,
            "effective_threshold": round(effective_threshold, 3)
        }
