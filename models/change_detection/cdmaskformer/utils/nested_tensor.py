import torch
from torch import Tensor
from typing import List, Optional


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Convert a list of tensors to a nested tensor structure.
    Used for handling variable sized inputs in the loss computation.
    
    Args:
        tensor_list: list of tensors with possibly different sizes
        
    Returns:
        A NestedTensor with:
            - tensors: a tensor containing the per-tensor data
            - mask: a binary mask with 1 indicating valid regions
    """
    # Determine max size
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    
    # Determine batch shape
    batch_shape = [len(tensor_list)] + max_size
    b, c, h, w = batch_shape
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    
    # Create output tensor and mask
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], :img.shape[2]] = False
    
    return tensor, mask


def _max_by_axis(the_list):
    """
    Find maximum size along each dimension across a list of lists/tuples.
    
    Args:
        the_list: list of lists/tuples
        
    Returns:
        list with maximum sizes
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for idx, item in enumerate(sublist):
            maxes[idx] = max(maxes[idx], item)
    return maxes 