import yaml
import logging
from typing import Dict, List, Union

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class StaticScoringEngine:
    def __init__(self, rule_file: str = "configs/static_rules.yaml"):
        self.rule_file = rule_file
        self.rules = self._load_rules()
        logger.info("✅ StaticScoringEngine initialized.")

    def _load_rules(self) -> Dict[str, List[Dict]]:
        try:
            with open(self.rule_file, encoding="utf-8") as f:
                rules = yaml.safe_load(f)
                logger.info(f"📜 Loaded rules from {self.rule_file}")
                return rules
        except Exception as e:
            logger.error(f"❌ Failed to load static rules: {e}")
            return {}

    def compute_score(self, features: Dict[str, Union[float, int, str]]) -> Dict[str, Union[float, bool, str, List]]:
        total_score = 0.0
        details = []
        auto_reject = False
        stop_remark = None

        for feature, rules in self.rules.items():
            value = features.get(feature, 0)
            applied = False

            for rule in rules:
                score_delta = rule.get("score", 0)
                remark = rule.get("remark", "")
                reject = rule.get("reject", False) or "Стоп фактор" in remark
                passed = False

                if "value" in rule:
                    passed = value == rule["value"]
                elif "min" in rule and "max" in rule:
                    passed = rule["min"] <= value <= rule["max"]
                elif "min" in rule:
                    passed = value >= rule["min"]
                elif "max" in rule:
                    passed = value <= rule["max"]

                if passed:
                    total_score += score_delta
                    details.append({
                        "feature": feature,
                        "value": value,
                        "rule": rule,
                        "score_delta": score_delta,
                        "remark": remark
                    })
                    if reject:
                        auto_reject = True
                        stop_remark = remark
                        logger.warning(f"🚫 STOP FACTOR triggered on {feature}: {remark}")
                    applied = True
                    break

            if not applied:
                details.append({
                    "feature": feature,
                    "value": value,
                    "rule": "No matching rule",
                    "score_delta": 0,
                    "remark": ""
                })

        return {
            "score": round(total_score, 2),
            "auto_reject": auto_reject,
            "stop_remark": stop_remark,
            "details": details
        }

    def score_applicant(self, features: Dict[str, Union[float, int, str]]) -> float:
        result = self.compute_score(features)
        normalized_score = min(max(result["score"] / 100.0, 0.0), 1.0)
        logger.info(f"📊 Static Score Applicant Normalized: {normalized_score:.4f}")
        return normalized_score
