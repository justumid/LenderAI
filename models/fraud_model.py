import torch
from typing import List, Dict, Any, Optional
from models.vae import FraudVAE


class FraudModel:
    """
    Wrapper around FraudVAE to expose standardized scoring interface.
    """

    def __init__(
        self,
        input_dim: int = 8,
        sequence_length: int = 24,
        model_path: Optional[str] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = FraudVAE(input_dim=input_dim, seq_len=sequence_length)
        self.model.to(self.device)
        self.model.eval()

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded FraudVAE weights from {model_path}")

    def predict(self, sequence: List[List[float]]) -> float:
        """
        Inputs:
            sequence: List of 24 time steps × input_dim floats

        Output:
            Anomaly score (higher = more suspicious), float in [0, 1]
        """
        tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            score = self.model.anomaly_score(tensor).item()

        return round(float(score), 4)

    def batch_predict(self, sequences: List[List[List[float]]]) -> List[float]:
        """
        Optional batch mode.
        Inputs:
            sequences: List of [T x D] sequences

        Returns:
            List of fraud scores
        """
        tensor = torch.tensor(sequences, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            scores = self.model.anomaly_score(tensor)

        return [round(s.item(), 4) for s in scores]
