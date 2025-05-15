# app/services/review_decision_service.py

import datetime
import logging
from typing import Optional, List, Dict

from data_pipeline.data_ingestion import get_db_connection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ReviewDecisionService:
    """
    Service to handle manual review decisions.
    - Save approve/reject
    - Load history
    - Prepare feedback for retraining
    """

    def __init__(self):
        logger.info("ReviewDecisionService initialized.")

    def save_review(
        self,
        pinfl: str,
        decision: bool,
        reviewer: str,
        review_notes: Optional[str] = None
    ) -> None:
        """
        Save (or update) a manual review decision.
        :param pinfl: Applicant's PINFL
        :param decision: True = approve, False = reject
        :param reviewer: Who made the decision
        :param review_notes: Optional free text
        """
        timestamp = datetime.datetime.utcnow()

        query = """
            INSERT INTO manual_reviews (pinfl, review_decision, review_notes, reviewer, review_timestamp)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (pinfl) DO UPDATE
            SET
                review_decision = EXCLUDED.review_decision,
                review_notes = EXCLUDED.review_notes,
                reviewer = EXCLUDED.reviewer,
                review_timestamp = EXCLUDED.review_timestamp;
        """

        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (pinfl, int(decision), review_notes, reviewer, timestamp))
            conn.commit()
            logger.info(f"Manual review saved for PINFL={pinfl}: {'Approved' if decision else 'Rejected'} by {reviewer}")
        except Exception as e:
            logger.error(f"Error saving review for PINFL={pinfl}: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()

    def load_review(self, pinfl: str) -> Optional[Dict[str, any]]:
        """
        Load a manual review decision for a given PINFL.
        """
        query = """
            SELECT pinfl, review_decision, review_notes, reviewer, review_timestamp
            FROM manual_reviews
            WHERE pinfl = %s
            LIMIT 1;
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, (pinfl,))
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "pinfl": row[0],
                    "decision": bool(row[1]),
                    "notes": row[2],
                    "reviewer": row[3],
                    "timestamp": row[4]
                }
        finally:
            conn.close()

    def list_recent_reviews(self, limit: int = 100) -> List[Dict[str, any]]:
        """
        List most recent manual reviews.
        """
        query = f"""
            SELECT pinfl, review_decision, review_notes, reviewer, review_timestamp
            FROM manual_reviews
            ORDER BY review_timestamp DESC
            LIMIT {limit};
        """
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                return [
                    {
                        "pinfl": row[0],
                        "decision": bool(row[1]),
                        "notes": row[2],
                        "reviewer": row[3],
                        "timestamp": row[4]
                    }
                    for row in rows
                ]
        finally:
            conn.close()

    def prepare_feedback_samples(self) -> List[Dict[str, any]]:
        """
        Prepare manually reviewed samples for model retraining feedback.
        Could be used later by retraining pipelines.
        """
        reviews = self.list_recent_reviews(limit=10000)
        feedback_samples = []
        for r in reviews:
            feedback_samples.append({
                "pinfl": r["pinfl"],
                "target_label": int(r["decision"]),  # 1 = approve, 0 = reject
                "review_timestamp": r["timestamp"]
            })
        logger.info(f"Prepared {len(feedback_samples)} feedback samples for retraining.")
        return feedback_samples


if __name__ == "__main__":
    # Example usage
    service = ReviewDecisionService()

    # Save a review
    service.save_review(
        pinfl="30303901621130",
        decision=True,
        reviewer="admin_user",
        review_notes="Looks fine after salary verification."
    )

    # Load review
    review = service.load_review("30303901621130")
    print("Loaded review:", review)

    # List recent
    recent = service.list_recent_reviews(limit=5)
    print("Recent reviews:", recent)

    # Prepare feedback samples
    feedback = service.prepare_feedback_samples()
    print(f"Feedback samples count: {len(feedback)}")
