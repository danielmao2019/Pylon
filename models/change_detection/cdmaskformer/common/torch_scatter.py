"""
Pure PyTorch implementations of scatter operations.
These are equivalent to torch_scatter operations but implemented using native PyTorch.
"""

import torch
from typing import Optional, Union, Tuple

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter operation that adds values from src into out at the indices specified by index.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor indicating where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of the output tensor along scatter dimension. If None, will be inferred from index.
        
    Returns:
        Output tensor with scattered and added values
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter operation that computes mean of values from src at the indices specified by index.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor indicating where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of the output tensor along scatter dimension. If None, will be inferred from index.
        
    Returns:
        Output tensor with scattered and averaged values
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=torch.long, device=src.device)
    
    # Count occurrences of each index
    count.index_add_(0, index, torch.ones_like(index, dtype=torch.long))
    
    # Add values
    out.index_add_(0, index, src)
    
    # Compute mean
    count = count.unsqueeze(-1).expand_as(out)
    return out / count.clamp(min=1)

def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter operation that computes maximum of values from src at the indices specified by index.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor indicating where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of the output tensor along scatter dimension. If None, will be inferred from index.
        
    Returns:
        Tuple of (output tensor with scattered and maximized values, argmax indices)
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    out = torch.full((dim_size, *src.shape[1:]), float('-inf'), dtype=src.dtype, device=src.device)
    argmax = torch.zeros(dim_size, *src.shape[1:], dtype=torch.long, device=src.device)
    
    # For each unique index
    unique_indices = torch.unique(index)
    for idx in unique_indices:
        mask = index == idx
        values = src[mask]
        max_val, max_idx = torch.max(values, dim=0)
        out[idx] = max_val
        argmax[idx] = max_idx
        
    return out, argmax

def scatter_min(src: torch.Tensor, index: torch.Tensor, dim: int = 0, dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scatter operation that computes minimum of values from src at the indices specified by index.
    
    Args:
        src: Source tensor to scatter
        index: Index tensor indicating where to scatter
        dim: Dimension along which to scatter
        dim_size: Size of the output tensor along scatter dimension. If None, will be inferred from index.
        
    Returns:
        Tuple of (output tensor with scattered and minimized values, argmin indices)
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
        
    out = torch.full((dim_size, *src.shape[1:]), float('inf'), dtype=src.dtype, device=src.device)
    argmin = torch.zeros(dim_size, *src.shape[1:], dtype=torch.long, device=src.device)
    
    # For each unique index
    unique_indices = torch.unique(index)
    for idx in unique_indices:
        mask = index == idx
        values = src[mask]
        min_val, min_idx = torch.min(values, dim=0)
        out[idx] = min_val
        argmin[idx] = min_idx
        
    return out, argmin 