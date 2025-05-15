import torch
from typing import List, Dict, Union, Optional, Any
from models.financial_bert import FinancialBERT
from models.sequence_encoder import SequenceEncoder


class BERTSequencePDModel:
    """
    Production wrapper for FinancialBERT.
    Supports:
    - Sequence padding and masking
    - CPU/GPU inference
    - Full risk output (PD, LGD, EAD, limit)
    - Embedding return for downstream use
    """

    def __init__(
        self,
        input_dim: int = 8,
        max_seq_len: int = 24,
        model_path: Optional[str] = None,
        pretrained_model: Optional[torch.nn.Module] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.encoder = SequenceEncoder(input_dim=input_dim, max_seq_len=max_seq_len)

        if pretrained_model:
            self.model = pretrained_model
        else:
            self.model = FinancialBERT(input_dim=input_dim)

        self.model.to(self.device)
        self.model.eval()

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ Loaded BERT model from {model_path}")

    def predict(
        self,
        sequence: List[List[float]],
        return_full: bool = False
    ) -> Union[float, Dict[str, Any]]:
        """
        Run inference for a single sequence.
        Returns PD by default or full model outputs.
        """
        x, mask = self.encoder.encode(sequence)
        x = x.unsqueeze(0).to(self.device)         # [1, T, D]
        mask = mask.unsqueeze(0).to(self.device)   # [1, T]

        with torch.no_grad():
            output = self.model(x, mask)

        if not return_full:
            return round(output["pd"].squeeze().item(), 4)

        return {
            "pd": round(output["pd"].squeeze().item(), 4),
            "lgd": round(output["lgd"].squeeze().item(), 4),
            "ead": round(output["ead"].squeeze().item(), 4),
            "loan_limit": round(output["loan_limit"].squeeze().item(), 2),
            "embedding": output["embedding"].squeeze().cpu().tolist()
        }

    def batch_predict(
        self,
        sequences: List[List[List[float]]],
        return_full: bool = False
    ) -> Union[List[float], List[Dict[str, Any]]]:
        """
        Run inference on a batch of sequences.
        """
        xs, masks = self.encoder.batch_encode(sequences)
        xs = xs.to(self.device)
        masks = masks.to(self.device)

        with torch.no_grad():
            output = self.model(xs, masks)

        if not return_full:
            return [round(v.item(), 4) for v in output["pd"]]

        return [
            {
                "pd": round(output["pd"][i].item(), 4),
                "lgd": round(output["lgd"][i].item(), 4),
                "ead": round(output["ead"][i].item(), 4),
                "loan_limit": round(output["loan_limit"][i].item(), 2),
                "embedding": output["embedding"][i].cpu().tolist()
            }
            for i in range(xs.size(0))
        ]
