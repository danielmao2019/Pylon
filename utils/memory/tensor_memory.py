"""Utilities for measuring GPU memory usage of PyTorch tensors and tensor dictionaries."""

import torch
from typing import Dict, Any


def get_tensor_memory(tensor: torch.Tensor) -> Dict[str, Any]:
    """Return memory info for a single tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dict with keys:
          - 'bytes': total bytes
          - 'mib': size in MiB
          - 'device': 'GPU' if cuda else 'CPU'
          - 'shape': tuple shape
          - 'dtype': string dtype name
    """
    assert isinstance(tensor, torch.Tensor), "get_tensor_memory expects a torch.Tensor"
    tensor_bytes = tensor.element_size() * tensor.numel()
    return {
        'bytes': tensor_bytes,
        'mib': tensor_bytes / (1024**2),
        'device': 'GPU' if tensor.is_cuda else 'CPU',
        'shape': tuple(tensor.shape),
        'dtype': str(tensor.dtype),
    }


def get_pc_dict_memory(pc_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Get CPU and GPU memory usage of point cloud dictionary in MiB.

    Args:
        pc_dict: Dictionary containing PyTorch tensors (e.g., point cloud data)

    Returns:
        Dictionary containing:
        - 'gpu_mib': GPU memory usage in MiB
        - 'cpu_mib': CPU memory usage in MiB
        - 'total_mib': Total memory usage in MiB
        - 'gpu_tensors': Number of GPU tensors
        - 'cpu_tensors': Number of CPU tensors
        - 'details': Dictionary with per-key memory usage and device info
    """
    gpu_bytes = 0
    cpu_bytes = 0
    details = {}
    gpu_tensor_count = 0
    cpu_tensor_count = 0

    for key, tensor in pc_dict.items():
        if isinstance(tensor, torch.Tensor):
            info = get_tensor_memory(tensor)
            if tensor.is_cuda:
                gpu_bytes += info['bytes']
                gpu_tensor_count += 1
            else:
                cpu_bytes += info['bytes']
                cpu_tensor_count += 1
            details[key] = {k: info[k] for k in ['mib', 'device']}
        else:
            # Non-tensor values
            details[key] = {'mib': 0.0, 'device': 'N/A'}

    total_bytes = gpu_bytes + cpu_bytes

    return {
        'gpu_mib': gpu_bytes / (1024**2),
        'cpu_mib': cpu_bytes / (1024**2),
        'total_mib': total_bytes / (1024**2),
        'gpu_tensors': gpu_tensor_count,
        'cpu_tensors': cpu_tensor_count,
        'details': details,
    }
