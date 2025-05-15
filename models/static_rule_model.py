import yaml
import os
import logging
from typing import Dict, Union, Tuple, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StaticRuleEngine:
    """
    Lightweight rule-based scoring engine.
    Designed for testing, fallback scoring, or offline batch scoring.
    """

    def __init__(self, rule_file: str = "configs/static_rules.yaml"):
        self.rule_file = rule_file
        self.rules = self._load_rules()

    def _load_rules(self) -> Dict:
        if not os.path.exists(self.rule_file):
            logger.warning(f"[StaticRuleEngine] Rule file not found: {self.rule_file}")
            return {}
        try:
            with open(self.rule_file, "r", encoding="utf-8") as f:
                rules = yaml.safe_load(f)
            logger.info(f"[StaticRuleEngine] ✅ Loaded rules from {self.rule_file}")
            return rules
        except Exception as e:
            logger.error(f"[StaticRuleEngine] ❌ Failed to load YAML: {e}")
            return {}

    def score(self, features: Dict[str, Union[int, float, str]]) -> float:
        score, _, _ = self.score_with_details(features)
        return score

    def score_with_details(self, features: Dict[str, Union[int, float, str]]) -> Tuple[float, bool, List[str]]:
        """
        Computes total score and explains rule hits.
        Returns:
            normalized_score: float in [0, 1]
            auto_reject: bool
            remarks: List of matched rule descriptions
        """
        total_score = 0.0
        max_possible_score = 0.0
        matched_remarks = []
        auto_reject = False

        for field, conditions in self.rules.items():
            if field not in features:
                continue
            value = features[field]

            for rule in conditions:
                passed = self._check_condition(value, rule)
                score_delta = rule.get("score", 0.0)
                max_possible_score += score_delta

                if passed:
                    total_score += score_delta
                    if "remark" in rule:
                        matched_remarks.append(rule["remark"])
                        if "Стоп фактор" in rule["remark"] or rule.get("reject", False):
                            auto_reject = True
                    break  # apply first matching rule

        norm_score = round(total_score / max_possible_score, 4) if max_possible_score > 0 else 0.0
        return norm_score, auto_reject, matched_remarks

    def _check_condition(self, value: Union[int, float, str], rule: Dict) -> bool:
        try:
            if "value" in rule:
                return value == rule["value"]
            if "min" in rule and "max" in rule:
                return rule["min"] <= value <= rule["max"]
            elif "min" in rule:
                return value >= rule["min"]
            elif "max" in rule:
                return value <= rule["max"]
        except Exception as e:
            logger.warning(f"[Rule Check] Failed condition check: {e}")
        return False
