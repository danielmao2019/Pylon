from typing import Tuple, Any
import torch


def check_semantic_segmentation_pred(tensor: Any) -> torch.Tensor:
    assert type(tensor) == torch.Tensor, f"{type(tensor)=}"
    assert tensor.dim() == 4, f"{tensor.shape=}"
    assert tensor.is_floating_point(), f"{tensor.dtype=}"
    return tensor


def check_semantic_segmentation_true(tensor: Any) -> torch.Tensor:
    assert type(tensor) == torch.Tensor, f"{type(tensor)=}"
    assert tensor.dim() == 3, f"{tensor.shape=}"
    assert tensor.dtype == torch.int64, f"{tensor.dtype=}"
    return tensor


def check_semantic_segmentation(y_pred: Any, y_true: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    check_semantic_segmentation_pred(tensor=y_pred)
    check_semantic_segmentation_true(tensor=y_true)
    assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true


# ====================================================================================================
# depth estimation
# ====================================================================================================


def check_depth_estimation_pred(tensor: Any) -> torch.Tensor:
    assert type(tensor) == torch.Tensor, f"{type(tensor)=}"
    assert tensor.dim() == 4 and tensor.shape[1] == 1, f"{tensor.shape=}"
    assert tensor.is_floating_point(), f"{tensor.dtype=}"
    return tensor


def check_depth_estimation_true(tensor: Any) -> torch.Tensor:
    assert type(tensor) == torch.Tensor, f"{type(tensor)=}"
    assert tensor.dim() == 3, f"{tensor.shape=}"
    assert tensor.is_floating_point(), f"{tensor.dtype=}"
    assert tensor.min() >= 0, f"{tensor.min()=}"
    return tensor


def check_depth_estimation(y_pred: Any, y_true: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    check_depth_estimation_pred(tensor=y_pred)
    check_depth_estimation_true(tensor=y_true)
    assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
    assert y_pred.shape[-2:] == y_true.shape[-2:], f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true


# ====================================================================================================
# normal estimation
# ====================================================================================================


def check_normal_estimation_pred(tensor: Any) -> torch.Tensor:
    assert type(tensor) == torch.Tensor, f"{type(tensor)=}"
    assert tensor.dim() == 4 and tensor.shape[1] == 3, f"{tensor.shape=}"
    assert tensor.is_floating_point(), f"{tensor.dtype=}"
    return tensor


def check_normal_estimation_true(tensor: Any) -> torch.Tensor:
    return check_normal_estimation_pred(tensor=tensor)


def check_normal_estimation(y_pred: Any, y_true: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    check_normal_estimation_pred(tensor=y_pred)
    check_normal_estimation_true(tensor=y_true)
    assert y_pred.shape == y_true.shape, f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true
