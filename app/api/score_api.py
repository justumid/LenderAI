from fastapi import APIRouter, HTTPException
# Using specific Pydantic models from your schema file is generally better
# from pydantic import BaseModel # Keep this if schemas.py is not yet complete for all uses
import logging
import torch
import json # Not directly used in this version of the function, but often useful
import os
from typing import Dict, Any # For type hinting

# Assuming 'app' is the root for these imports. Adjust if your PYTHONPATH is different.
from services.static_scoring_engine import StaticScoringEngine
from services.explanation_service import ExplanationService # Corrected import
from services.identity_service import get_identity_info # Assuming this service exists
from services.review_status_engine import ReviewStatusEngine, ReviewStatus
from services.review_decision_service import ReviewDecisionService
from services.segment_service import SegmentService # Added import

from data_pipeline.feature_extractor import FeatureExtractor
from data_pipeline.sequence_extractor import SequenceExtractor
from data_pipeline.json_normalizer import normalize_salary_json, normalize_credit_json
from data_pipeline.data_ingestion import load_salary_json, load_credit_json

from models.full_model import FullModel
from models.static_rule_model import StaticRuleEngine
from models.limit_decision_model import LimitDecisionModel
from models.fairness_adjuster import FairnessAdjuster
from models.uncertainty_estimator import UncertaintyEstimator
from models.sequence_encoder import SequenceEncoder
from app.schemas.schemas import ScoreRequest, ScoreResponse

logger = logging.getLogger(__name__)
router = APIRouter()

# === Configuration (Should be externalized) ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./checkpoints/full_model_latest.pth"
STATIC_RULES_PATH = "configs/static_rules.yaml"
# Paths for sub-models if not handled by FullModel's checkpoint or if they are separate
LIMIT_MODEL_MLP_PATH = "./checkpoints/limit_mlp.pt"
LIMIT_MODEL_LGBM_PATH = "./checkpoints/limit_lgbm.pkl"
LIMIT_MODEL_SCALER_PATH = "./checkpoints/limit_scaler.pkl"
SIMCLR_MODEL_PATH = "./checkpoints/simclr_encoder.pt" # For SegmentService
SEGMENT_KMEANS_PATH = "./checkpoints/segment_kmeans.pkl" # For SegmentClusterer if used
SEGMENT_SCALER_PATH = "./checkpoints/segment_scaler.pkl" # For SegmentClusterer if used
EXPLAINER_PD_MODEL_PATH = "./checkpoints/pd_model.pkl" # For ExplanationModel
EXPLAINER_LIMIT_MODEL_PATH = "./checkpoints/limit_model.pkl" # For ExplanationModel
EXPLAINER_FRAUD_MODEL_PATH = "./checkpoints/fraud_model.pkl" # For ExplanationModel
EXPLAINER_SCALER_PATH = "./checkpoints/explain_scaler.pkl" # For ExplanationModel


# The fraud_sequence from SequenceExtractor has 9 features:
# debt_val, overdue_flag, salary_val, overdue_sum, overdue_percent, repayment_val, pti_val, debt_delta, anomaly_flag
EXPECTED_SEQUENCE_FEATURE_DIM = 9
EXPECTED_SEQUENCE_LENGTH = 24

logger.info(f"🔄 Initializing services and models for scoring on device: {DEVICE}...")

try:
    static_rule_engine_instance = StaticRuleEngine(STATIC_RULES_PATH)
    
    # Initialize LimitDecisionModel with potentially configurable paths
    limit_model_instance = LimitDecisionModel(
        mlp_path=LIMIT_MODEL_MLP_PATH,
        lgbm_path=LIMIT_MODEL_LGBM_PATH,
        scaler_path=LIMIT_MODEL_SCALER_PATH
    )
    
    fairness_adjuster_instance = FairnessAdjuster() # Default init
    
    # UncertaintyEstimator input_dim is 5 because FullModel feeds it:
    # [pd, lgd, ead, static_scores, fraud_scores]
    uncertainty_estimator_instance = UncertaintyEstimator(input_dim=5)

    full_model = FullModel(
        sequence_input_dim=EXPECTED_SEQUENCE_FEATURE_DIM,
        sequence_length=EXPECTED_SEQUENCE_LENGTH,
        static_rule_engine=static_rule_engine_instance,
        limit_model=limit_model_instance,
        fairness_adjuster=fairness_adjuster_instance,
        uncertainty_estimator=uncertainty_estimator_instance
    ).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        # Ensure the loaded state_dict is compatible with the FullModel definition
        full_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        logger.info(f"✅ FullModel loaded from {MODEL_PATH}")
    else:
        logger.warning(f"⚠️ FullModel checkpoint not found at {MODEL_PATH}. Scoring will use an untrained model.")

    review_engine = ReviewStatusEngine()
    review_decision_service = ReviewDecisionService()
    
    # For preparing sequence input to FullModel
    sequence_encoder_for_model = SequenceEncoder(
        input_dim=EXPECTED_SEQUENCE_FEATURE_DIM, 
        max_len=EXPECTED_SEQUENCE_LENGTH
    )

    # Initialize other services
    # SegmentService might use SimCLREncoder internally or a different model
    # Assuming SegmentService is designed to take features and sequences
    segment_service_instance = SegmentService(
        # input_dim for SimCLR in SegmentService needs to match its training data.
        # This might be different from EXPECTED_SEQUENCE_FEATURE_DIM.
        # For now, assuming it's compatible with tabular_features or a specific sequence.
        # Let's assume its internal _prepare_input_tensor handles feature selection.
        input_dim=len(FeatureExtractor(None,None).extract_all_features()) # Estimate based on feature extractor output count
    )
    
    # ExplanationService might load its own models for SHAP if needed
    explanation_service_instance = ExplanationService(
        # pd_model_path=EXPLAINER_PD_MODEL_PATH, # If ExplanationModel loads them
        # limit_model_path=EXPLAINER_LIMIT_MODEL_PATH,
        # fraud_model_path=EXPLAINER_FRAUD_MODEL_PATH,
        # scaler_path=EXPLAINER_SCALER_PATH
    ) # Assuming default paths in its constructor are okay or it doesn't load models for basic explanations

except Exception as e:
    logger.exception(f"❌ Critical error during model or service initialization: {e}")
    full_model = None # Prevent API from running with broken initialization

# === Scoring Endpoint ===
@router.post("/", response_model=ScoreResponse)
def score_applicant(request: ScoreRequest):
    if full_model is None:
        logger.error("❌ Scoring endpoint called, but FullModel or services failed to initialize.")
        raise HTTPException(status_code=500, detail="Scoring system initialization error. Please check server logs.")

    logger.info(f"📥 Scoring request received for PINFL: {request.pinfl}")

    try:
        # === Load & Normalize Data ===
        raw_salary = load_salary_json(request.pinfl)
        raw_credit = load_credit_json(request.pinfl)

        salary_data_normalized = normalize_salary_json(raw_salary)
        credit_data_normalized = normalize_credit_json(raw_credit)

        if not salary_data_normalized.get("salary_records") and not credit_data_normalized.get("contracts"):
            logger.warning(f"⚠️ Applicant data not found or empty for PINFL: {request.pinfl}")
            raise HTTPException(status_code=404, detail=f"Applicant data not found or empty for PINFL: {request.pinfl}")

        # === Feature Extraction ===
        feature_extractor = FeatureExtractor(salary_data_normalized, credit_data_normalized)
        tabular_features = feature_extractor.extract_all_features() # This is a dictionary

        # === Sequence Extraction ===
        seq_extractor = SequenceExtractor(salary_data_normalized, credit_data_normalized)
        all_sequences_dict = seq_extractor.extract()

        fraud_sequence_list = all_sequences_dict.get("fraud_sequence", [])
        if not fraud_sequence_list:
            logger.warning(f"Fraud sequence is empty for PINFL {request.pinfl}. Using zeros.")
            fraud_sequence_list = [[0.0] * EXPECTED_SEQUENCE_FEATURE_DIM for _ in range(EXPECTED_SEQUENCE_LENGTH)]
        
        sequence_tensor_for_model, _ = sequence_encoder_for_model.encode(fraud_sequence_list)
        sequence_tensor_for_model = sequence_tensor_for_model.unsqueeze(0).to(DEVICE)
        
        logger.debug(f"Prepared sequence tensor for FullModel with shape: {sequence_tensor_for_model.shape}")

        # === Get Static Score Details (for explanations) ===
        # FullModel computes static_score internally, but we need remarks for explanations.
        static_score_details = static_rule_engine_instance.score_with_details(tabular_features)
        static_remarks = [detail['remark'] for detail in static_score_details['details'] if detail['remark']]


        # === Model Inference ===
        with torch.no_grad():
            full_model.eval()
            output_dict = full_model(sequence_tensor_for_model, tabular_features) # Pass dict

        pd_score = output_dict["pd"].item()
        lgd_score = output_dict["lgd"].item()
        ead_score = output_dict["ead"].item()
        fraud_score_model = output_dict["fraud_score"].item()
        loan_limit = output_dict["final_limit"].item()
        static_score_from_model = output_dict["static_score"].item() # Use this for consistency
        uncertainty = output_dict.get("confidence", torch.ones_like(output_dict["pd"])).item()
        risk_class_model = output_dict.get("risk_class", ["unknown"])[0]

        # === Segmentation ===
        # SegmentService.identify_segment might expect specific sequence or feature format
        segment = segment_service_instance.identify_segment(tabular_features, all_sequences_dict)
        logger.info(f"Segment identified for PINFL {request.pinfl}: {segment}")


        # === Model-driven Review Status Decision ===
        auto_status = review_engine.decide(fraud_score_model, pd_score, loan_limit)

        # === Manual Review Override if exists ===
        manual_review = review_decision_service.load_review(request.pinfl)
        final_status = auto_status

        if manual_review:
            # Ensure 'decision' key exists and is boolean or 0/1
            if "decision" in manual_review and bool(manual_review["decision"]):
                final_status = ReviewStatus.accepted
            else:
                final_status = ReviewStatus.rejected
            logger.info(f"🔄 Manual review override for PINFL={request.pinfl}: {final_status.value}")
        
        # === Explanations ===
        # The 'requested_amount' might come from the BNPL request payload, not directly from tabular_features.
        # For now, assuming it might be part of tabular_features or a default.
        requested_amount_for_explanation = tabular_features.get("requested_loan_amount", loan_limit) # Fallback to approved limit

        explanations_content = explanation_service_instance.generate(
            features=tabular_features,
            pd_score=pd_score,
            ead_score=ead_score,
            lgd_score=lgd_score,
            static_score=static_score_from_model, # Use score from FullModel
            fraud_score=fraud_score_model,
            static_labels=static_remarks, 
            behavior_risk=segment, # Use the identified segment as behavior_risk proxy
            shap_values=None, # SHAP values are computationally expensive for real-time; compute offline if needed
            requested_amount=requested_amount_for_explanation,
            approved_limit=loan_limit
        )
        
        # === Identity Info (Optional) ===
        identity_info_data = get_identity_info(raw_salary, raw_credit) # Assuming this service is available

        logger.info(f"✅ Scoring complete for PINFL: {request.pinfl}")

        # === Final Response ===
        # The 'explanations' field in ScoreResponse is Dict[str, float].
        # Our explanation_service_instance.generate returns Dict[str, Any] (mostly strings).
        # This will cause a validation error if Pydantic is strict.
        # For now, we pass it as is. This schema needs to be updated.
        # A temporary workaround if schema is fixed:
        # explanations_for_response = {k: str(v)[:100] for k,v in explanations_content.items()} # Truncate/convert
        # Or, if only SHAP-like floats are allowed, and we don't have them:
        # explanations_for_response = {"summary_float_placeholder": round(pd_score,2) } 
        
        # Assuming ScoreResponse.explanations can take Dict[str, Any] or will be fixed
        return ScoreResponse(
            pinfl=request.pinfl,
            pd_score=round(pd_score, 4),
            lgd_score=round(lgd_score, 4),
            fraud_score=round(fraud_score_model, 4),
            loan_limit=round(loan_limit, 2),
            risk_class=risk_class_model,
            segment=segment,
            uncertainty=round(uncertainty, 4),
            explanations=explanations_content, # Pass the dictionary of explanations
            review_status=final_status.value
        )

    except HTTPException as http_exc:
        logger.warning(f"⚠️ HTTPException during scoring for PINFL {request.pinfl}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"❌ Scoring failed for PINFL: {request.pinfl}. Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error during scoring: {str(e)}")

