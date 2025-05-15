class MonthlyPaymentLimitCalculator:
    def __init__(self, salary: float, pd_score: float, fraud_score: float,
                 segment: str = "standard", client_flags: dict = None, static_rules: dict = None):
        """
        Parameters:
        - salary: Client's monthly income (UZS).
        - pd_score: Probability of Default (0.0 - 1.0).
        - fraud_score: Fraud anomaly score (0.0 - 1.0).
        - segment: Client segment (trusted, standard, risky, new).
        - client_flags: Dict of binary flags (e.g., has_property, has_stable_job).
        - static_rules: Optional dict with business-specific rules.
        """
        self.salary = salary
        self.pd_score = pd_score
        self.fraud_score = fraud_score
        self.segment = segment
        self.client_flags = client_flags or {}
        self.static_rules = static_rules or self.default_static_rules()

    def default_static_rules(self):
        return {
            "base_affordability_cap": 0.3,
            "segment_caps": {
                "trusted": 0.4,
                "standard": 0.3,
                "new": 0.25,
                "risky": 0.2
            },
            "pd_penalty_weight": 0.5,
            "fraud_penalty_weight": 0.5,
            "property_bonus": 0.05,
            "stable_job_bonus": 0.05,
            "min_limit_ratio": 0.1,
            "cap_floor": 0.15,
            "cap_ceiling": 0.5
        }

    def base_cap(self):
        return self.static_rules["segment_caps"].get(self.segment, self.static_rules["base_affordability_cap"])

    def compute_risk_adjustments(self):
        pd_penalty = self.pd_score * self.static_rules["pd_penalty_weight"]
        fraud_penalty = self.fraud_score * self.static_rules["fraud_penalty_weight"]
        return pd_penalty, fraud_penalty

    def compute_positive_adjustments(self):
        property_bonus = self.static_rules["property_bonus"] if self.client_flags.get("has_property") else 0.0
        stable_job_bonus = self.static_rules["stable_job_bonus"] if self.client_flags.get("has_stable_job") else 0.0
        return property_bonus + stable_job_bonus

    def effective_cap(self):
        base = self.base_cap()
        pd_penalty, fraud_penalty = self.compute_risk_adjustments()
        positive_bonus = self.compute_positive_adjustments()

        risk_factor = 1.0 - (pd_penalty + fraud_penalty)
        risk_factor = max(risk_factor, 0.0)

        cap = base * risk_factor + positive_bonus

        cap = max(cap, self.static_rules["cap_floor"])
        cap = min(cap, self.static_rules["cap_ceiling"])

        return cap

    def calculate_monthly_limit(self):
        cap = self.effective_cap()
        limit = self.salary * cap
        return round(limit, 2)

    def get_details(self):
        pd_penalty, fraud_penalty = self.compute_risk_adjustments()
        positive_bonus = self.compute_positive_adjustments()

        return {
            "salary": self.salary,
            "pd_score": self.pd_score,
            "fraud_score": self.fraud_score,
            "segment": self.segment,
            "base_cap": self.base_cap(),
            "pd_penalty": pd_penalty,
            "fraud_penalty": fraud_penalty,
            "positive_bonus": positive_bonus,
            "final_effective_cap": self.effective_cap(),
            "monthly_payment_limit": self.calculate_monthly_limit()
        }
