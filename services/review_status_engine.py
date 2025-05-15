import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ReviewStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    manual_review = "manual_review"

class ReviewStatusEngine:
    """
    Simple rule-based engine to decide review status based on model outputs.
    """

    def __init__(self):
        logger.info("ReviewStatusEngine initialized.")

    def decide(
        self,
        fraud_score: float,
        pd_score: float,
        loan_limit: float
    ) -> ReviewStatus:
        """
        Decide review status based on fraud_score, pd_score, loan_limit.
        """
        logger.debug(f"Evaluating review status: fraud={fraud_score}, pd={pd_score}, limit={loan_limit}")

        if fraud_score > 0.7 or pd_score > 0.5:
            return ReviewStatus.manual_review

        if loan_limit == 0:
            return ReviewStatus.rejected

        if fraud_score < 0.3 and pd_score < 0.2:
            return ReviewStatus.accepted

        return ReviewStatus.pending
