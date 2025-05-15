# data_pipeline/utils.py

from typing import List

def pad_sequence(seq: List[List[float]], max_len: int, pad_val: float = 0.0) -> List[List[float]]:
    """
    Pads a sequence of vectors to a fixed length with the given pad value.
    Ensures all vectors have consistent dimension.
    """
    if not seq:
        return [[pad_val] * 1 for _ in range(max_len)]

    feature_dim = len(seq[0])
    seq = seq[:max_len]
    pad_len = max_len - len(seq)
    return seq + [[pad_val] * feature_dim for _ in range(pad_len)]
