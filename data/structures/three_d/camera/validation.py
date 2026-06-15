from typing import Any, Union

import numpy as np
import torch

from utils.ops.materialize_tensor import materialize_tensor

_ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS = 32


def validate_camera_convention(convention: Any) -> str:
    # Input validations
    assert isinstance(convention, str), f"{type(convention)=}"
    assert convention in [
        "standard",
        "opengl",
        "opencv",
        "pytorch3d",
        "arkit",
    ], f"Unsupported convention: {convention}"
    return convention


def validate_camera_intrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, np.ndarray):
        return _validate_camera_intrinsics_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return _validate_camera_intrinsics_torch(obj)
    raise TypeError(
        f"Camera intrinsics must be a numpy array or a torch tensor, got {type(obj)}"
    )


def _validate_camera_intrinsics_numpy(obj: Any) -> np.ndarray:
    # Input validations
    assert isinstance(obj, np.ndarray), f"{type(obj)=}"
    assert obj.ndim >= 2, f"{obj.ndim=}"
    assert obj.shape[-2:] == (3, 3), f"{obj.shape=}"
    assert obj.dtype == np.float32, f"{obj.dtype=}"

    fx = obj[..., 0, 0]
    fy = obj[..., 1, 1]
    assert np.all(fx > 0), "Focal length element [0, 0] must be positive."
    assert np.all(fy > 0), "Focal length element [1, 1] must be positive."
    cx = obj[..., 0, 2]
    cy = obj[..., 1, 2]
    assert np.all(cx >= 0), "Principal point element [0, 2] must be non-negative."
    assert np.all(cy >= 0), "Principal point element [1, 2] must be non-negative."
    assert np.all(obj[..., 0, 1] == 0), "Skew element [0, 1] must be 0."
    assert np.all(obj[..., 1, 0] == 0), "Skew element [1, 0] must be 0."
    expected_last_row = np.array([0, 0, 1], dtype=obj.dtype)
    assert np.all(
        obj[..., 2, :] == expected_last_row
    ), "Camera intrinsics must have [0, 0, 1] in the last row."
    return obj


def _validate_camera_intrinsics_torch(obj: Any) -> torch.Tensor:
    # Input validations
    assert isinstance(obj, torch.Tensor), f"{type(obj)=}"
    assert obj.ndim >= 2, f"{obj.ndim=}"
    assert obj.shape[-2:] == (3, 3), f"{obj.shape=}"
    assert obj.dtype == torch.float32, f"{obj.dtype=}"

    fx = obj[..., 0, 0]
    fy = obj[..., 1, 1]
    assert torch.all(fx > 0), "Focal length element [0, 0] must be positive."
    assert torch.all(fy > 0), "Focal length element [1, 1] must be positive."
    cx = obj[..., 0, 2]
    cy = obj[..., 1, 2]
    assert torch.all(cx >= 0), "Principal point element [0, 2] must be non-negative."
    assert torch.all(cy >= 0), "Principal point element [1, 2] must be non-negative."
    assert torch.all(obj[..., 0, 1] == 0), "Skew element [0, 1] must be 0."
    assert torch.all(obj[..., 1, 0] == 0), "Skew element [1, 0] must be 0."
    expected_last_row = torch.tensor(
        [0, 0, 1],
        dtype=obj.dtype,
        device=obj.device,
    )
    assert torch.all(
        obj[..., 2, :] == expected_last_row
    ), "Camera intrinsics must have [0, 0, 1] in the last row."
    return obj


def validate_camera_extrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, np.ndarray):
        return _validate_camera_extrinsics_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return _validate_camera_extrinsics_torch(obj)
    raise TypeError(
        f"Camera extrinsics must be a numpy array or a torch tensor, got {type(obj)}"
    )


def _validate_camera_extrinsics_numpy(obj: Any) -> np.ndarray:
    # Input validations
    assert isinstance(obj, np.ndarray), f"{type(obj)=}"
    assert obj.ndim >= 2, f"{obj.ndim=}"
    assert obj.shape[-2:] == (4, 4), f"{obj.shape=}"
    assert obj.dtype in (np.float32, np.float64), f"{obj.dtype=}"

    expected_last_row = np.array([0, 0, 0, 1], dtype=obj.dtype)
    assert np.allclose(
        obj[..., 3, :],
        expected_last_row,
        atol=0.0,
        rtol=0.0,
    ), "Camera extrinsics must have [0, 0, 0, 1] in the last row."
    rotation = obj[..., :3, :3]
    _validate_rotation_matrix_numpy(rotation)
    return obj


def _validate_camera_extrinsics_torch(obj: Any) -> torch.Tensor:
    # Input validations
    assert isinstance(obj, torch.Tensor), f"{type(obj)=}"
    assert obj.ndim >= 2, f"{obj.ndim=}"
    assert obj.shape[-2:] == (4, 4), f"{obj.shape=}"
    assert obj.dtype in (torch.float32, torch.float64), f"{obj.dtype=}"

    expected_last_row = torch.tensor(
        [0, 0, 0, 1],
        dtype=obj.dtype,
        device=obj.device,
    )
    assert torch.allclose(
        obj[..., 3, :],
        expected_last_row,
        atol=0.0,
        rtol=0.0,
    ), "Camera extrinsics must have [0, 0, 0, 1] in the last row."
    rotation = obj[..., :3, :3]
    _validate_rotation_matrix_torch(rotation)
    return obj


def validate_rotation_matrix(obj: Any) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, np.ndarray):
        return _validate_rotation_matrix_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return _validate_rotation_matrix_torch(obj)
    raise TypeError(
        f"Rotation matrix must be a numpy array or a torch tensor, got {type(obj)}"
    )


def _validate_rotation_matrix_numpy(obj: Any) -> np.ndarray:
    # Input validations
    assert isinstance(obj, np.ndarray), f"{type(obj)=}"
    assert obj.ndim >= 2, f"{obj.ndim=}"
    assert obj.shape[-2:] == (3, 3), f"{obj.shape=}"
    assert obj.dtype in (np.float32, np.float64), f"{obj.dtype=}"

    atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(
        np.finfo(np.float32).eps
    )
    atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(
        np.finfo(np.float64).eps
    )
    if obj.dtype == np.float32:
        return _validate_rotation_matrix_numpy_against_threshold(
            obj, threshold=atol_float32
        )
    if obj.dtype == np.float64:
        return _validate_rotation_matrix_numpy_against_threshold(
            obj, threshold=atol_float64
        )
    assert 0, "should not reach here."


def _validate_rotation_matrix_torch(obj: Any) -> torch.Tensor:
    # Input validations
    assert isinstance(obj, torch.Tensor), f"{type(obj)=}"
    assert obj.ndim >= 2, f"{obj.ndim=}"
    assert obj.shape[-2:] == (3, 3), f"{obj.shape=}"
    assert obj.dtype in (torch.float32, torch.float64), f"{obj.dtype=}"

    atol_float32 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(
        torch.finfo(torch.float32).eps
    )
    atol_float64 = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(
        torch.finfo(torch.float64).eps
    )
    if obj.dtype == torch.float32:
        return _validate_rotation_matrix_torch_against_threshold(
            obj, threshold=atol_float32
        )
    if obj.dtype == torch.float64:
        return _validate_rotation_matrix_torch_against_threshold(
            obj, threshold=atol_float64
        )
    assert 0, "should not reach here."


def _validate_rotation_matrix_numpy_against_threshold(
    obj: np.ndarray, threshold: float
) -> np.ndarray:
    identity = np.eye(3, dtype=obj.dtype)
    should_be_identity = obj @ np.swapaxes(obj, -1, -2)
    max_diff = float(np.max(np.abs(should_be_identity - identity)))
    assert np.allclose(
        should_be_identity,
        identity,
        atol=threshold,
        rtol=0.0,
    ), "Rotation matrix must be orthogonal. Max diff between RR^T and I: {:.6g} (threshold={:.6g})".format(
        max_diff, threshold
    )

    det = np.linalg.det(obj)
    assert np.allclose(
        det,
        1.0,
        atol=threshold,
        rtol=0.0,
    ), f"Rotation matrix must have determinant +1. det(R) = {det} (threshold={threshold})"

    return obj


def _validate_rotation_matrix_torch_against_threshold(
    obj: torch.Tensor, threshold: float
) -> torch.Tensor:
    # Input normalizations
    obj = materialize_tensor(obj)

    identity = torch.eye(3, dtype=obj.dtype, device=obj.device)
    should_be_identity = obj @ obj.transpose(-1, -2)
    max_diff = torch.max(torch.abs(should_be_identity - identity))
    assert torch.allclose(
        should_be_identity,
        identity,
        atol=threshold,
        rtol=0.0,
    ), (
        "Rotation matrix must be orthogonal. Max diff between RR^T and I: "
        f"{float(max_diff)} (threshold={threshold})"
    )

    det = torch.linalg.det(obj)
    ones = torch.ones_like(det)
    assert torch.allclose(
        det,
        ones,
        atol=threshold,
        rtol=0.0,
    ), f"Rotation matrix must have determinant +1. det(R) = {det} (threshold={threshold})"

    return obj
