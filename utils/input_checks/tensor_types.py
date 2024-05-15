from typing import Tuple, Any, Optional
import torch


def check_image(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.dim() == (4 if batched else 3) and obj.shape[-3] == 3, f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    return obj


# ====================================================================================================
# semantic segmentation
# ====================================================================================================


def check_semantic_segmentation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.dim() == (4 if batched else 3), f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    return obj


def check_semantic_segmentation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.dim() == (3 if batched else 2), f"{obj.shape=}"
    assert obj.dtype == torch.int64, f"{obj.dtype=}"
    return obj


def check_semantic_segmentation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    check_semantic_segmentation_pred(obj=y_pred, batched=batched)
    check_semantic_segmentation_true(obj=y_true, batched=batched)
    if batched:
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
    return y_pred, y_true


# ====================================================================================================
# depth estimation
# ====================================================================================================


def check_depth_estimation_pred(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.dim() == (4 if batched else 3) and obj.shape[-3] == 1, f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    return obj


def check_depth_estimation_true(obj: Any, batched: Optional[bool] = True) -> torch.Tensor:
    assert type(obj) == torch.Tensor, f"{type(obj)=}"
    assert obj.dim() == (3 if batched else 2), f"{obj.shape=}"
    assert obj.is_floating_point(), f"{obj.dtype=}"
    assert obj.min() >= 0, f"{obj.min()=}"
    return obj


def check_depth_estimation(y_pred: Any, y_true: Any, batched: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor]:
    check_depth_estimation_pred(obj=y_pred, batched=batched)
    check_depth_estimation_true(obj=y_true, batched=batched)
    if batched:
        assert y_pred.shape[0] == y_true.shape[0], f"{y_pred.shape=}, {y_true.shape=}"
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
