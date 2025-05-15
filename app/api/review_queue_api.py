# app/api/review_queue_api.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
import time

from data_pipeline.review_data_handler import (
    load_review,
    load_all_reviews,
    save_review,
    delete_review  # 🔥 add delete support
)

router = APIRouter()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Request Models ---

class ReviewSubmitRequest(BaseModel):
    pinfl: str
    review_decision: bool  # True = approve, False = reject
    review_notes: Optional[str] = None

# --- Endpoints ---

@router.get("/review/list")
async def list_reviews():
    """
    List all manual reviews stored.
    """
    try:
        reviews = load_all_reviews()
        return {"reviews": reviews}
    except Exception as e:
        logger.error(f"Failed to load reviews: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while loading reviews.")

@router.get("/review/get/{pinfl}")
async def get_review(pinfl: str):
    """
    Get review for a specific applicant PINFL.
    """
    try:
        review = load_review(pinfl)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found for this applicant.")
        return {"review": review}
    except Exception as e:
        logger.error(f"Error retrieving review for PINFL {pinfl}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@router.post("/review/submit")
async def submit_review(request: ReviewSubmitRequest):
    """
    Submit or update a manual review decision.
    """
    try:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        review_record = {
            "pinfl": request.pinfl,
            "review_decision": request.review_decision,
            "review_notes": request.review_notes,
            "timestamp": timestamp
        }
        save_review(**review_record)
        logger.info(f"✅ Review submitted for PINFL {request.pinfl}")
        return {"message": "Review submitted successfully.", "timestamp": timestamp}
    except Exception as e:
        logger.error(f"Failed to submit review: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during review submission.")

@router.delete("/review/delete/{pinfl}")
async def delete_review_record(pinfl: str):
    """
    Delete (cancel) a manual review decision.
    """
    try:
        success = delete_review(pinfl)
        if not success:
            raise HTTPException(status_code=404, detail="No review found to delete.")
        logger.info(f"✅ Review deleted for PINFL {pinfl}")
        return {"message": f"Review for PINFL {pinfl} deleted successfully."}
    except Exception as e:
        logger.error(f"Failed to delete review: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during review deletion.")
