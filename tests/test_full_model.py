import torch
import random
from models.full_model import FullModel
from models.static_rule_model import StaticRuleEngine
from models.limit_decision_model import LimitDecisionModel
from models.fairness_adjuster import FairnessAdjuster
from models.uncertainty_estimator import UncertaintyEstimator


def generate_dummy_static_features(batch_size: int):
    return {
        "salary_mean": torch.tensor([random.uniform(2e6, 10e6) for _ in range(batch_size)]),
        "avg_payment_to_income_ratio": torch.tensor([random.uniform(0.2, 0.8) for _ in range(batch_size)]),
        "repayment_consistency_ratio": torch.tensor([random.uniform(0.7, 1.0) for _ in range(batch_size)]),
        "open_overdue_ratio": torch.tensor([random.uniform(0.0, 0.4) for _ in range(batch_size)]),
        "credit_burden_index": torch.tensor([random.uniform(0.2, 0.9) for _ in range(batch_size)]),
        "katm_score": torch.tensor([random.randint(300, 500) for _ in range(batch_size)]),
        "gender_female": torch.tensor([random.choice([0, 1]) for _ in range(batch_size)]),  # for fairness adjuster
    }


def test_full_model(batch_size=4, seq_len=24, input_dim=8):
    print("🔍 Initializing FullModel...")

    static_engine = StaticRuleEngine("configs/static_rules.yaml")
    limit_model = LimitDecisionModel()
    fairness = FairnessAdjuster(adjustment_rules={"gender_female": 0.05})
    uncertainty = UncertaintyEstimator(input_dim=5)

    model = FullModel(
        sequence_input_dim=input_dim,
        sequence_length=seq_len,
        static_rule_engine=static_engine,
        limit_model=limit_model,
        fairness_adjuster=fairness,
        uncertainty_estimator=uncertainty,
        use_static_score=True
    )

    sequence_tensor = torch.rand(batch_size, seq_len, input_dim)
    static_features = generate_dummy_static_features(batch_size)

    print("🚀 Running forward pass...")
    output = model(sequence_tensor, static_features)

    # === Assertions
    expected_keys = [
        "pd", "lgd", "ead",
        "fraud_score", "fraud_flag",
        "static_score", "bert_limit", "final_limit",
        "final_score", "weighted_score",
        "risk_class", "confidence"
    ]

    for key in expected_keys:
        assert key in output, f"❌ Missing key in output: {key}"

    print("✅ All keys present in output.")

    # === Sample Output
    print("\n📊 Sample Outputs:")
    print("📉 PD:", output["pd"].detach().cpu().numpy())
    print("🎯 Final Score:", output["final_score"].detach().cpu().numpy())
    print("🏦 Loan Limit (BERT):", output["bert_limit"].detach().cpu().numpy())
    print("🛠 Loan Limit (Final):", output["final_limit"].detach().cpu().numpy())
    print("🚩 Fraud Flag:", output["fraud_flag"].detach().cpu().numpy())
    print("📊 Risk Class:", output["risk_class"])
    print("🔐 Confidence:", output["confidence"].detach().cpu().numpy())


if __name__ == "__main__":
    test_full_model()
