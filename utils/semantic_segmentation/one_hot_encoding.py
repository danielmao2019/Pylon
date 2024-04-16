from typing import Optional
import torch


def to_one_hot_encoding(mask: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None) -> torch.Tensor:
    r"""This function assumes mask is of shape (N, H, W) or (H, W) and transforms
    mask into binary tensors of shape (N, C, H, W) and (C, H, W), respectively.
    """
    # input checks
    assert mask.dim() in [2, 3], f"{mask.shape=}"
    assert mask.dtype == torch.int64, f"{mask.dtype=}"
    assert type(num_classes) == int, f"{type(num_classes)=}"
    assert ignore_index is None or type(ignore_index) == int, f"{type(ignore_index)=}"
    # transform into one-hot encoding
    new_mask = mask.clone()
    new_mask = new_mask.unsqueeze(-3)
    target_size = torch.tensor(new_mask.shape, dtype=torch.int64)
    if ignore_index is not None:
        assert num_classes not in new_mask, f"{num_classes=}, {new_mask.unique()=}"
        new_mask[new_mask == ignore_index] = num_classes
        target_size[-3] = target_size[-3] * (num_classes + 1)
    else:
        target_size[-3] = target_size[-3] * num_classes
    result = torch.zeros(size=tuple(target_size), dtype=torch.int64, device=new_mask.device)
    result = result.scatter_(-3, new_mask, 1)
    result = result[..., :num_classes, :, :]
    # sanity check
    assert new_mask.shape[-3] == 1, f"{new_mask.shape=}"
    assert result.dim() == new_mask.dim()
    assert result.shape[-2:] == new_mask.shape[-2:]
    assert result.shape[-3] == num_classes
    return result
