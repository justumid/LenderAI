import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from typing import Dict, Union

class PDHead(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.head(x)

class PDEstimator:
    def __init__(self, encoder_model="bert-base-uncased",
                       head_checkpoint="./checkpoints/pd_head.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pretrained BERT encoder
        self.tokenizer = BertTokenizer.from_pretrained(encoder_model)
        self.encoder = BertModel.from_pretrained(encoder_model).to(self.device)
        
        # Load trained PD head
        self.head = PDHead().to(self.device)
        self.head.load_state_dict(torch.load(head_checkpoint, map_location=self.device))

        self.encoder.eval()
        self.head.eval()

    def predict_with_model(self, sequences: Dict[str, Union[list, float]]) -> float:
        credit_sequence = sequences.get("credit_sequence", [])
        pti_sequence = sequences.get("pti_sequence", [])
        text_sequence = " ".join([str(x) for x in credit_sequence + pti_sequence])

        inputs = self.tokenizer(text_sequence, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            pooled_output = outputs.pooler_output  # [B, 768]
            pd_prob = self.head(pooled_output).squeeze().item()

        pd_prob = 0.01 + pd_prob * 0.99
        return round(pd_prob, 4)

    def predict_with_rules(self, features: Dict[str, Union[float, int]]) -> float:
        # (This will reuse your rule-based logic unchanged — as fallback)
        base_pd = 0.03
        pti = features.get("avg_payment_to_income_ratio", 0.5)
        dti = features.get("dti_ratio", 0.5)
        consistency = features.get("repayment_consistency_ratio", 0.8)
        volatility = features.get("repayment_volatility", 0.3)
        overdue_ratio = features.get("open_overdue_ratio", 0.1)
        overdue_qty = features.get("overview_overdue_principal_qty", 0)
        salary = features.get("salary_mean", 0.0)
        salary_growth = features.get("salary_growth", 0.0)
        katm = features.get("katm_score", 400)
        num_requests = features.get("num_credit_requests", 10)
        num_loans = features.get("active_loans_count", 2)
        fraud_score = features.get("fraud_score", 0.1)
        early_repayments = features.get("num_early_repayments", 0)
        segment = features.get("risk_segment", "mid")

        # Affordability stress
        if pti > 0.6 or dti > 0.6: base_pd += 0.15
        elif pti > 0.4: base_pd += 0.07

        if consistency < 0.6: base_pd += 0.12
        elif consistency < 0.8: base_pd += 0.06
        elif consistency >= 0.95: base_pd -= 0.03

        if volatility > 0.4: base_pd += 0.05
        elif volatility < 0.1: base_pd -= 0.02

        if overdue_ratio > 0.3: base_pd += 0.12
        elif overdue_ratio > 0.1: base_pd += 0.06

        if overdue_qty >= 5: base_pd += 0.1
        if salary < 2_000_000: base_pd += 0.1
        elif salary > 10_000_000: base_pd -= 0.05

        if salary_growth < 0: base_pd += 0.03
        elif salary_growth > 0.2: base_pd -= 0.02

        if katm < 350: base_pd += 0.1
        elif katm >= 450: base_pd -= 0.05

        if num_requests >= 50: base_pd += 0.05
        elif num_requests == 0: base_pd += 0.02

        if num_loans >= 4: base_pd += 0.04
        if early_repayments >= 2: base_pd -= 0.04

        if fraud_score > 0.3: base_pd += 0.1
        elif fraud_score > 0.15: base_pd += 0.05

        segment_multiplier = {"high": 1.2, "mid": 1.0, "low": 0.85}.get(segment.lower(), 1.0)
        base_pd *= segment_multiplier

        return round(min(max(base_pd, 0.01), 1.0), 4)

    def predict(self, features: Dict[str, Union[float, int]], sequences: Dict[str, list]) -> float:
        try:
            return self.predict_with_model(sequences)
        except Exception as e:
            print(f"⚠️ Model prediction failed: {e} — using rule-based fallback")
            return self.predict_with_rules(features)
