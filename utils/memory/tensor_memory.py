"""Utilities for measuring GPU memory usage of PyTorch tensors and tensor dictionaries."""

import torch
from typing import Dict, Any


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
            tensor_bytes = tensor.element_size() * tensor.numel()
            tensor_mib = tensor_bytes / (1024**2)  # Convert to MiB
            
            if tensor.is_cuda:
                gpu_bytes += tensor_bytes
                gpu_tensor_count += 1
                details[key] = {'mib': tensor_mib, 'device': 'GPU'}
            else:
                cpu_bytes += tensor_bytes
                cpu_tensor_count += 1
                details[key] = {'mib': tensor_mib, 'device': 'CPU'}
        else:
            # Non-tensor values (shouldn't happen in point clouds but handle gracefully)
            details[key] = {'mib': 0.0, 'device': 'N/A'}
    
    total_bytes = gpu_bytes + cpu_bytes
    
    return {
        'gpu_mib': gpu_bytes / (1024**2),
        'cpu_mib': cpu_bytes / (1024**2),
        'total_mib': total_bytes / (1024**2),
        'gpu_tensors': gpu_tensor_count,
        'cpu_tensors': cpu_tensor_count,
        'details': details
    }