# app/services/behavior_analyzer.py

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BehaviorAnalyzer:
    """
    Analyzes behavioral patterns of the applicant:
    - Salary stability
    - Open contract behavior
    - Overdue and rejections behavior
    """

    def __init__(self):
        logger.info("✅ BehaviorAnalyzer initialized.")

    def assess_risk(self, features: Dict[str, Any]) -> str:
        """
        Assess overall behavioral risk level based on salary stability and credit behavior.
        Returns:
          - "Low"     (good applicant)
          - "Medium"  (some risks)
          - "High"    (risky behavior)
        """
        salary_risk = self._salary_stability(features)
        credit_risk = self._credit_behavior(features)

        if salary_risk == "Low" and credit_risk == "Low":
            return "Low"
        elif salary_risk == "High" or credit_risk == "High":
            return "High"
        else:
            return "Medium"

    def segment(self, features: Dict[str, Any]) -> str:
        """
        Segmentation for loan limit or strategy purposes.
        """
        return self.assess_risk(features)

    def _salary_stability(self, features: Dict[str, Any]) -> str:
        """
        Analyze salary history for stability.
        - Low risk if steady salary growth or low volatility
        - High risk if big drops or unstable salary
        """
        salaries: List[float] = features.get("salary_last_6_months", [])

        if not salaries or len(salaries) < 6:
            logger.info("⚠ Salary history insufficient for stability analysis.")
            return "Medium"

        salaries = [float(s) for s in salaries if s is not None]

        if len(salaries) < 6:
            return "High"

        mean_salary = sum(salaries) / len(salaries)
        std_salary = (sum((s - mean_salary) ** 2 for s in salaries) / len(salaries)) ** 0.5

        # Stability logic
        std_ratio = std_salary / (mean_salary + 1e-8)

        if std_ratio < 0.15:
            return "Low"      # Very stable
        elif std_ratio < 0.3:
            return "Medium"   # Acceptable
        else:
            return "High"     # Unstable salaries

    def _credit_behavior(self, features: Dict[str, Any]) -> str:
        """
        Analyze credit behavior:
        - Number of open contracts
        - Overdue debts
        - Rejection history
        """
        num_open_contracts = features.get("num_open_contracts", 0)
        total_overdue_debt = features.get("open_all_overdue_sum", 0.0)
        num_rejections = features.get("num_rejections", 0)

        # Logic for credit behavior
        if total_overdue_debt > 1_000_000 or num_rejections >= 3 or num_open_contracts >= 5:
            return "High"
        if total_overdue_debt > 0 or num_rejections >= 1:
            return "Medium"

        return "Low"


if __name__ == "__main__":
    # Quick test
    analyzer = BehaviorAnalyzer()

    sample_features = {
        "salary_last_6_months": [7_000_000, 7_100_000, 7_050_000, 7_000_000, 6_950_000, 7_000_000],
        "num_open_contracts": 2,
        "open_all_overdue_sum": 0,
        "num_rejections": 0
    }

    risk = analyzer.assess_risk(sample_features)
    print(f"Behavior risk assessment: {risk}")
