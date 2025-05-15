import logging
from typing import Dict, Any

from app.services.static_scoring_engine import StaticScoringEngine
from app.services.fraud_engine import detect_fraud
from app.services.segment_service import identify_segment
from app.services.loan_limit_engine import predict_limit
from app.services.risk_engine import predict_risk
from app.services.explanation_service import generate_explanations
from app.services.vae_anomaly_engine import compute_anomaly_score
from app.services.identity_service import get_identity_info
from app.services.limit_decision_wrapper import adjust_limit_per_product
from app.services.monthly_payment_limit_calculator import MonthlyPaymentLimitCalculator
from data_pipeline.feature_extractor import FeatureExtractor
from app.services.preprocessing import preprocess_applicant, PreprocessingError

logger = logging.getLogger(__name__)
scoring_engine = StaticScoringEngine("configs/static_rules.yaml")


def score_applicant(pinfl: str, product_price: float = None, merchant_profile: Dict[str, Any] = None) -> Dict[str, Any]:
    try:
        # === Step 1: Preprocessing ===
        tensors = preprocess_applicant(pinfl)
        if not tensors:
            raise PreprocessingError("Preprocessing failed for PINFL.")

        raw_salary = tensors["raw_salary"]
        raw_credit = tensors["raw_credit"]
        extractor = FeatureExtractor(raw_salary, raw_credit)
        features = extractor.extract_all_features()
        modern = extractor.extract_modern_indicators()

        logger.info(f"[PINFL={pinfl}] ✅ Features extracted.")

        # === Step 2: Static Scoring ===
        static_result = scoring_engine.compute_score({**features, **modern})
        static_score = static_result["score"]

        if static_result["auto_reject"]:
            logger.warning(f"[PINFL={pinfl}] ❌ Auto-reject: {static_result['stop_remark']}")
            return build_response(pinfl, auto_reject=True, static_score=static_score, static_result=static_result,
                                  identity=get_identity_info(raw_salary, raw_credit), modern=modern, features=features)

        # === Step 3: Model Scorings ===
        x_seq = tensors["x_seq"]
        additional = tensors["additional_features"]

        segment_id = identify_segment(x_seq)
        fraud_score, fraud_alert = detect_fraud(x_seq, additional, static_score)
        risk_score = predict_risk(x_seq, additional, static_score)
        global_limit = predict_limit(x_seq, additional, static_score)
        vae_anomaly = compute_anomaly_score(x_seq)

        logger.info(f"[PINFL={pinfl}] ✅ Model scores computed.")

        # === Step 4: Limit Decision Wrapper ===
        if product_price and merchant_profile:
            adjusted_limit = adjust_limit_per_product(global_limit, product_price, merchant_profile, fraud_score, risk_score)
        else:
            adjusted_limit = global_limit

        # === Step 5: Monthly Payment Limit ===
        salary = features.get("salary_avg", 0)
        calc = MonthlyPaymentLimitCalculator(
            salary=salary,
            pd_score=risk_score,
            fraud_score=fraud_score,
            segment="trusted" if segment_id == 1 else "standard"  # Example mapping
        )
        monthly_payment_limit = calc.calculate_monthly_limit()
        payment_limit_details = calc.get_details()

        logger.info(f"[PINFL={pinfl}] ✅ Payment capacity estimated.")

        # === Step 6: Final Score Fusion ===
        final_score = round(
            0.4 * static_score +
            0.3 * (1 - fraud_score) * 100 +
            0.3 * (1 - risk_score) * 100, 2
        )

        explanation = generate_explanations(static_score, risk_score, fraud_score, adjusted_limit)

        # === Step 7: Decision Reasoning ===
        decision = "approve"
        decision_reasons = []
        if fraud_alert:
            decision = "review"
            decision_reasons.append("Fraud Alert Triggered")
        if risk_score >= 0.6:
            decision_reasons.append("High Risk Score")
        if adjusted_limit < product_price:
            decision_reasons.append("Adjusted Limit Insufficient")
        if monthly_payment_limit < 500_000:  # Example threshold
            decision_reasons.append("Low Monthly Payment Capacity")

        # Final decision downgrade
        if static_result.get("soft_decline", False):
            decision = "review"
            decision_reasons.append(static_result["soft_decline_remark"])

        logger.info(f"[PINFL={pinfl}] ✅ Final score computed.")

        return {
            "pinfl": pinfl,
            "final_score": final_score,
            "loan_limit": adjusted_limit,
            "fraud_score": round(fraud_score, 3),
            "risk_score": round(risk_score, 3),
            "vae_anomaly": round(vae_anomaly, 3),
            "segment_id": int(segment_id),
            "fraud_alert": fraud_alert,
            "static_score": round(static_score, 2),
            "identity": get_identity_info(raw_salary, raw_credit),
            "modern_indicators": modern,
            "monthly_payment_limit": monthly_payment_limit,
            "payment_limit_details": payment_limit_details,
            "explanations": explanation,
            "raw_features": features,
            "static_rules": static_result["details"],
            "decision": decision,
            "decision_reasons": decision_reasons,
            "auto_reject": False,
            "stop_remark": None,
        }

    except PreprocessingError as e:
        logger.error(f"[PINFL={pinfl}] ❌ Preprocessing Error: {e}")
        return {"error": str(e), "pinfl": pinfl}

    except Exception as e:
        logger.exception(f"[PINFL={pinfl}] ❌ Unexpected error during scoring.")
        return {"error": "Internal error during scoring", "pinfl": pinfl}


def build_response(pinfl, auto_reject, static_score, static_result, identity, modern, features):
    return {
        "pinfl": pinfl,
        "final_score": 0,
        "loan_limit": 0,
        "fraud_score": None,
        "risk_score": None,
        "vae_anomaly": None,
        "segment_id": None,
        "fraud_alert": True,
        "static_score": static_score,
        "identity": identity,
        "modern_indicators": modern,
        "monthly_payment_limit": 0,
        "payment_limit_details": {},
        "explanations": [],
        "raw_features": features,
        "static_rules": static_result["details"],
        "decision": "reject",
        "decision_reasons": [static_result["stop_remark"]],
        "auto_reject": auto_reject,
        "stop_remark": static_result["stop_remark"],
    }
