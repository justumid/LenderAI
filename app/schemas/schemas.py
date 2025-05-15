from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# === Score API Schemas ===

class ScoreRequest(BaseModel):
    pinfl: str = Field(..., example="30202836860013")

class ScoreResponse(BaseModel):
    pinfl: str
    pd_score: float
    lgd_score: float
    fraud_score: float
    loan_limit: float
    risk_class: str
    segment: str
    uncertainty: float
    explanations: Dict[str, float]

# === Train API Schemas ===

class TrainRequest(BaseModel):
    dataset_path: str = Field(..., example="./demo_data/combined_dataset.json")
    epochs: int = Field(5, example=5, ge=1)
    small_dev: bool = Field(False, description="Run on small sample for debugging")

class TrainResponse(BaseModel):
    message: str
    status: str

# === Monitor API Schemas ===

class MonitorResponse(BaseModel):
    model_version: str
    drift_status: str
    drift_metric: float
    calibration_status: str
    calibration_metric: float
    model_health: str

# === Review Queue API Schemas ===

class ReviewItem(BaseModel):
    pinfl: str = Field(..., example="30202836860013")
    reason: str = Field(..., example="Fraud suspicion")
    status: str = Field("pending", example="pending", description="pending, reviewed, rejected")
    comment: Optional[str] = Field("", description="Reviewer's comment")

class ReviewResponse(BaseModel):
    message: str
    status: str

class ReviewListResponse(BaseModel):
    items: List[ReviewItem]
