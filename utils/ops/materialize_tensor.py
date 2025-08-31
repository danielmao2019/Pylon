"""Tensor materialization utilities to avoid lazy wrapper issues."""

import torch
from typing import Any


def materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a tensor to avoid lazy wrapper issues.
    
    This function ensures that a tensor is fully materialized in memory,
    which can be important when dealing with certain tensor operations
    that may involve lazy evaluation or wrapped tensors.
    
    Args:
        tensor: Input tensor that may be lazily evaluated or wrapped
        
    Returns:
        Materialized tensor with the same data but guaranteed to be
        fully evaluated and accessible
    """
    assert isinstance(tensor, torch.Tensor), f"Input must be torch.Tensor, got {type(tensor)}"
    
    # Simple approach: clone and detach to ensure materialization
    # This creates a new tensor with the same data but removes any
    # lazy evaluation wrappers or computation graph dependencies
    return tensor.clone().detach()
