from typing import Any, Dict, Tuple
import torch


def check_point_cloud(pc: Dict[str, torch.Tensor]) -> None:
    assert isinstance(pc, dict)
    assert all(isinstance(k, str) for k in pc.keys())
    assert all(isinstance(v, torch.Tensor) for v in pc.values())

    assert 'pos' in pc, f"{pc.keys()=}"
    check_pc_xyz(obj=pc['pos'])

    if 'rgb' in pc:
        check_pc_rgb(obj=pc['rgb'])

    length = pc['pos'].shape[0]
    assert all(pc[key].shape[0] == length for key in pc.keys() if key != 'pos'), \
        f"{{{', '.join(f'{k}: {pc[k].shape}' for k in pc.keys())}}}"

    device = pc['pos'].device
    assert all(pc[key].device == device for key in pc.keys() if key != 'pos'), \
        f"{{{', '.join(f'{k}: {pc[k].device}' for k in pc.keys())}}}"


def check_pc_xyz(obj: Any) -> None:
    assert isinstance(obj, torch.Tensor), f"Expected torch.Tensor, got {type(obj)}"
    assert obj.ndim == 2, f"Expected 2D tensor, got shape {obj.shape}"
    assert obj.shape[0] > 0, f"Expected positive number of points, got {obj.shape[0]}"
    assert obj.shape[1] == 3, f"Expected shape (N, 3), got {obj.shape}"
    assert obj.is_floating_point(), f"Expected floating point tensor, got {obj.dtype}"
    assert not torch.isnan(obj).any(), "Tensor contains NaN values"
    assert not torch.isinf(obj).any(), "Tensor contains Inf values"


def check_pc_rgb(obj: Any) -> None:
    """No guarantee if this is in int format or in float format.
    """
    assert isinstance(obj, torch.Tensor), f"Expected torch.Tensor, got {type(obj)}"
    assert obj.ndim == 2, f"Expected 2D tensor, got shape {obj.shape}"
    assert obj.shape[0] > 0, f"Expected positive number of points, got {obj.shape[0]}"
    assert obj.shape[1] == 3, f"Expected shape (N, 3), got {obj.shape}"
    assert not torch.isnan(obj).any(), "Tensor contains NaN values"
    assert not torch.isinf(obj).any(), "Tensor contains Inf values"

# ====================================================================================================
# point cloud segmentation
# ====================================================================================================

def check_point_cloud_segmentation_pred(obj: Any, batched: bool = True) -> torch.Tensor:
    """
    Check if the prediction for point cloud segmentation is valid.
    
    Args:
        obj: The prediction tensor to check.
        batched: Whether the tensor contains points from multiple examples.
                 This parameter is kept for API consistency but doesn't affect validation,
                 as point clouds are handled as concatenated points.
        
    Returns:
        The validated tensor.
    
    Raises:
        AssertionError: If the tensor is not valid.
    """
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    # For all point cloud data, we expect shape [N, C] 
    # where N is the number of points and C is the number of classes
    assert obj.ndim == 2, f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_point_cloud_segmentation_true(obj: Any, batched: bool = True) -> torch.Tensor:
    """
    Check if the ground truth for point cloud segmentation is valid.
    
    Args:
        obj: The ground truth tensor to check.
        batched: Whether the tensor contains points from multiple examples.
                 This parameter is kept for API consistency but doesn't affect validation,
                 as point clouds are handled as concatenated points.
        
    Returns:
        The validated tensor.
    
    Raises:
        AssertionError: If the tensor is not valid.
    """
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    # For all point cloud data, we expect shape [N]
    # where N is the number of points
    assert obj.ndim == 1, f"{obj.shape=}"
    assert obj.dtype == torch.int64, f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_point_cloud_segmentation(y_pred: Any, y_true: Any, batched: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check if the prediction and ground truth for point cloud segmentation are valid.
    
    Args:
        y_pred: The prediction tensor to check.
        y_true: The ground truth tensor to check.
        batched: Whether the tensors contain points from multiple examples.
                 This parameter is kept for API consistency but doesn't affect validation,
                 as point clouds are handled as concatenated points.
        
    Returns:
        The validated tensors (y_pred, y_true).
    
    Raises:
        AssertionError: If the tensors are not valid.
    """
    check_point_cloud_segmentation_pred(obj=y_pred, batched=batched)
    check_point_cloud_segmentation_true(obj=y_true, batched=batched)
    # Check that the number of points match between prediction and ground truth
    assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true
