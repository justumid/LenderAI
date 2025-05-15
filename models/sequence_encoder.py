import torch
import numpy as np
from typing import List, Dict, Any, Tuple


class SequenceEncoder:
    """
    Pads and formats time-series sequences for Transformer/VAE input.
    Supports salary, repayments, overdue, etc.

    Usage:
        encoder = SequenceEncoder(input_dim=8, max_seq_len=24)
        tensor, mask = encoder.encode(sequences["repayments"])
    """

    def __init__(self, input_dim: int = 8, max_seq_len: int = 24):
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len

    def encode(self, sequence: List[List[float]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads a single sequence to fixed length.
        Returns:
            tensor: shape [max_seq_len, input_dim]
            mask: shape [max_seq_len], with 1 = valid, 0 = padded
        """
        if not sequence:
            sequence = [[0.0] * self.input_dim]

        # Truncate if too long
        sequence = sequence[-self.max_seq_len :]

        # Pad if too short
        padded = sequence + [[0.0] * self.input_dim] * (self.max_seq_len - len(sequence))

        tensor = torch.tensor(padded, dtype=torch.float32)
        mask = torch.tensor([1] * len(sequence) + [0] * (self.max_seq_len - len(sequence)), dtype=torch.long)

        return tensor, mask

    def batch_encode(self, sequences: List[List[List[float]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch-encodes multiple sequences.
        Returns:
            tensor: shape [B, max_seq_len, input_dim]
            mask: shape [B, max_seq_len]
        """
        batch_tensors = []
        batch_masks = []

        for seq in sequences:
            tensor, mask = self.encode(seq)
            batch_tensors.append(tensor)
            batch_masks.append(mask)

        return torch.stack(batch_tensors), torch.stack(batch_masks)

    def numpy_encode(self, sequence: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Numpy-compatible version for ML models or storage.
        Returns:
            X: np.ndarray [max_seq_len, input_dim]
            M: np.ndarray [max_seq_len]
        """
        tensor, mask = self.encode(sequence)
        return tensor.numpy(), mask.numpy()
