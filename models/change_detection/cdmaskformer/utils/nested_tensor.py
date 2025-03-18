import torch
from torch import Tensor
from typing import List, Optional


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Convert a list of tensors to a nested tensor structure.
    Used for handling variable sized inputs in the loss computation.
    
    Args:
        tensor_list: list of tensors with possibly different sizes
        
    Returns:
        A NestedTensor object with:
            - tensors: a tensor containing the per-tensor data
            - mask: a binary mask with False indicating valid regions, True for invalid/padded regions
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
    
    return NestedTensor(tensor, mask)


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