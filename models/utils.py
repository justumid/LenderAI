import torch
import numpy as np
import random
import os
import psutil
import datetime
from typing import Any, Dict, List, Optional



def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_numpy(tensor) -> np.ndarray:
    """
    Safely convert a PyTorch tensor to a NumPy array.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across NumPy, Torch, and Random.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_to_device(batch, device: torch.device):
    """
    Moves batch of tensors to device (list, tuple, or tensor).
    """
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    return batch.to(device)


def print_model_summary(model: torch.nn.Module, name: str = "Model"):
    """
    Print model structure and trainable parameters.
    """
    total_params = count_parameters(model)
    size_mb = sum(p.element_size() * p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n🧠 {name} Summary:")
    print(model)
    print(f"Total trainable parameters: {total_params:,} ({size_mb:.2f} MB)\n")


def log_memory_usage(tag: str = "Memory") -> None:
    """
    Log current CPU and GPU memory usage with timestamp.
    """
    now = datetime.datetime.now().strftime('%H:%M:%S')
    process = psutil.Process(os.getpid())
    mem_cpu = process.memory_info().rss / (1024 * 1024)

    msg = f"🧠 [{now}] {tag}: CPU = {mem_cpu:.2f} MB"

    if torch.cuda.is_available():
        mem_gpu = torch.cuda.memory_allocated() / (1024 * 1024)
        max_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
        msg += f", GPU = {mem_gpu:.2f} MB (Max = {max_gpu:.2f} MB)"

    print(msg)


def get_tensor_size(tensor: torch.Tensor) -> float:
    """
    Returns the size of a tensor in MB.
    """
    return tensor.element_size() * tensor.nelement() / (1024 ** 2)
