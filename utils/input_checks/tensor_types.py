from typing import Tuple, Any, Optional
import torch


def check_image(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (4 if batched else 3) and obj.shape[-3] == 3, f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj

# ====================================================================================================
# classification
# ====================================================================================================

def check_classification_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (2 if batched else 1), f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_classification_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (1 if batched else 0), f"{obj.shape=}"
    assert obj.dtype == torch.int64, f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_classification(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    check_classification_pred(obj=y_pred, batched=batched)
    check_classification_true(obj=y_true, batched=batched)
    if batched:
        assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}"
    # Check that y_true values don't exceed the number of classes
    # For classification, class dimension is at index 1 if batched, 0 if not
    num_classes = y_pred.size(1 if batched else 0)  # Class dimension depends on batched flag
    assert torch.all(y_true < num_classes), f"Found class indices >= {num_classes} in y_true. {y_pred.shape=}, {y_true.unique()=}."
    return y_pred, y_true

# ====================================================================================================
# semantic segmentation
# ====================================================================================================

def check_semantic_segmentation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (4 if batched else 3), f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_semantic_segmentation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (3 if batched else 2), f"{obj.shape=}"
    assert obj.dtype == torch.int64, f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_semantic_segmentation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    check_semantic_segmentation_pred(obj=y_pred, batched=batched)
    check_semantic_segmentation_true(obj=y_true, batched=batched)
    if batched:
        assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true

# ====================================================================================================
# depth estimation
# ====================================================================================================

def check_depth_estimation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (4 if batched else 3) and obj.shape[-3] == 1, f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_depth_estimation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (3 if batched else 2), f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert obj.min() >= 0, f"{obj.min()=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_depth_estimation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    check_depth_estimation_pred(obj=y_pred, batched=batched)
    check_depth_estimation_true(obj=y_true, batched=batched)
    if batched:
        assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}"
    assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true

# ====================================================================================================
# normal estimation
# ====================================================================================================

def check_normal_estimation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    return check_image(obj=obj, batched=batched)


def check_normal_estimation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    return check_image(obj=obj, batched=batched)


def check_normal_estimation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    check_normal_estimation_pred(obj=y_pred, batched=batched)
    check_normal_estimation_true(obj=y_true, batched=batched)
    assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true

# ====================================================================================================
# instance segmentation
# ====================================================================================================

def check_instance_segmentation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    """
    Check if the prediction for instance segmentation is valid.
    
    Args:
        obj: The prediction tensor to check.
        batched: Whether the tensor contains instances from multiple examples.
        
    Returns:
        The validated tensor.
    
    Raises:
        AssertionError: If the tensor is not valid.
    """
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (3 if batched else 2), f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_instance_segmentation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    """
    Check if the ground truth for instance segmentation is valid.
    
    Args:
        obj: The ground truth tensor to check.
        batched: Whether the tensor contains instances from multiple examples.
        
    Returns:
        The validated tensor.
    
    Raises:
        AssertionError: If the tensor is not valid.
    """
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.ndim == (3 if batched else 2), f"{obj.shape=}"
    assert obj.dtype == torch.int64, f"{obj.dtype=}"
    assert not torch.any(torch.isnan(obj))
    return obj


def check_instance_segmentation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Check if the prediction and ground truth for instance segmentation are valid.
    
    Args:
        y_pred: The prediction tensor to check.
        y_true: The ground truth tensor to check.
        batched: Whether the tensors contain instances from multiple examples.
        
    Returns:
        The validated tensors (y_pred, y_true).
    
    Raises:
        AssertionError: If the tensors are not valid.
    """
    check_instance_segmentation_pred(obj=y_pred, batched=batched)
    check_instance_segmentation_true(obj=y_true, batched=batched)
    if batched:
        assert y_pred.size(0) == y_true.size(0), f"{y_pred.shape=}, {y_true.shape=}"
    assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true

# ====================================================================================================
# point cloud segmentation
# ====================================================================================================

def check_point_cloud_segmentation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
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


def check_point_cloud_segmentation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
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


def check_point_cloud_segmentation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
