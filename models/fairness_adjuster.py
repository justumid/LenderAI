import numpy as np
import torch
import logging
from typing import List, Dict, Union, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class FairnessAdjuster:
    """
    Enterprise-Grade FairnessAdjuster:
    - Audits and mitigates group bias via multiple techniques:
        - Statistical Parity Difference
        - Disparate Impact Ratio (80% rule)
        - Equalized Odds (TPR, FPR)
        - Reweighting corrections
        - Reject Option Classification (ROC)
    """

    def __init__(self, correction_threshold=0.05, hard_flag_threshold=0.15, target_disparate_ratio=0.8):
        self.correction_threshold = correction_threshold
        self.hard_flag_threshold = hard_flag_threshold
        self.target_disparate_ratio = target_disparate_ratio
        self.stats = {}

    def audit(self, scores: List[float], groups: List[Dict[str, Any]], sensitive: str, y_true: List[int] = None):
        """
        Full audit of fairness metrics across sensitive feature.
        """
        group_scores = {}
        for score, meta in zip(scores, groups):
            group = str(meta.get(sensitive, "unknown"))
            group_scores.setdefault(group, []).append(score)

        group_medians = {g: np.median(v) for g, v in group_scores.items()}
        overall_median = np.median(scores)
        shifts = {g: overall_median - m for g, m in group_medians.items()}

        # Statistical Parity Difference
        group_means = {g: np.mean(v) for g, v in group_scores.items()}
        ref_group = next(iter(group_means))
        parity_diff = {g: group_means[g] - group_means[ref_group] for g in group_means}

        # Disparate Impact Ratio
        ratios = {g: group_means[g] / group_means[ref_group] if group_means[ref_group] else 1.0 for g in group_means}

        # Equalized Odds if labels provided
        eq_odds = {}
        if y_true:
            eq_odds = self._equalized_odds(y_true, scores, groups, sensitive)

        self.stats[sensitive] = {
            "medians": group_medians,
            "means": group_means,
            "shifts": shifts,
            "parity_diff": parity_diff,
            "disparate_impact": ratios,
            "equalized_odds": eq_odds
        }

        logger.info(f"Audit for {sensitive}: {self.stats[sensitive]}")
        return self.stats[sensitive]

    def _equalized_odds(self, y_true: List[int], y_pred: List[float], groups: List[Dict[str, Any]], sensitive: str):
        metrics = {}
        for g in set(meta.get(sensitive, "unknown") for meta in groups):
            idx = [i for i, meta in enumerate(groups) if meta.get(sensitive) == g]
            if not idx: continue
            true_g = np.array([y_true[i] for i in idx])
            pred_g = np.array([y_pred[i] >= 0.5 for i in idx])

            tpr = (np.sum((pred_g == 1) & (true_g == 1)) / max(1, np.sum(true_g == 1)))
            fpr = (np.sum((pred_g == 1) & (true_g == 0)) / max(1, np.sum(true_g == 0)))

            metrics[g] = {"TPR": round(tpr, 3), "FPR": round(fpr, 3)}
        return metrics

    def reweight(self, scores: torch.Tensor, sensitive_tensor: torch.Tensor, desired_ratio=1.0) -> torch.Tensor:
        """
        Soft re-weighting to achieve desired disparate impact ratio.
        """
        group_1_idx = (sensitive_tensor == 1).nonzero(as_tuple=True)[0]
        group_0_idx = (sensitive_tensor == 0).nonzero(as_tuple=True)[0]

        mean_1 = scores[group_1_idx].mean().item() if len(group_1_idx) > 0 else 0.0
        mean_0 = scores[group_0_idx].mean().item() if len(group_0_idx) > 0 else 0.0

        current_ratio = mean_1 / mean_0 if mean_0 else 1.0
        correction_factor = desired_ratio / current_ratio if current_ratio != 0 else 1.0

        adjusted_scores = scores.clone()
        adjusted_scores[group_1_idx] = (scores[group_1_idx] * correction_factor).clamp(0.0, 1.0)

        logger.info(f"Reweighted sensitive group (current_ratio={current_ratio:.3f}, correction={correction_factor:.3f})")
        return adjusted_scores

    def reject_option_adjustment(self, scores: torch.Tensor, groups: List[Dict[str, Any]], sensitive: str, reject_zone=(0.3, 0.7)):
        """
        Reject Option Classification (ROC):
        - Adjusts scores for borderline applicants to improve fairness.
        """
        adjusted_scores = scores.clone()
        for i, meta in enumerate(groups):
            s = scores[i].item()
            if reject_zone[0] < s < reject_zone[1]:
                if meta.get(sensitive) == 1:
                    adjusted_scores[i] = min(s + 0.05, 1.0)
                else:
                    adjusted_scores[i] = max(s - 0.05, 0.0)
        logger.info(f"Applied Reject Option Classification for sensitive={sensitive}")
        return adjusted_scores

    def apply_all(self, scores: torch.Tensor, groups: List[Dict[str, Any]], sensitive: str, y_true: List[int]):
        """
        Full audit + adjust pipeline:
        - Audit metrics
        - Reweight
        - ROC adjust
        """
        audit_stats = self.audit(scores.tolist(), groups, sensitive, y_true)
        sensitive_tensor = torch.tensor([1 if g.get(sensitive) == 1 else 0 for g in groups], dtype=torch.float32)

        scores = self.reweight(scores, sensitive_tensor, desired_ratio=self.target_disparate_ratio)
        scores = self.reject_option_adjustment(scores, groups, sensitive)

        return scores, audit_stats
