from typing import Any, Union
import numpy as np
import torch


def _check_camera_intrinsics_numpy(obj: Any) -> np.ndarray:
    assert isinstance(obj, np.ndarray), "Camera intrinsics must be a numpy array."
    assert obj.ndim == 2, "Camera intrinsics must be a 2D array."
    assert obj.shape == (3, 3), "Camera intrinsics must be of shape (3, 3)."
    assert obj.dtype in [
        np.float32,
        np.float64,
    ], "Camera intrinsics must be of type float32 or float64."
    fx, fy = obj[0, 0], obj[1, 1]
    assert (
        fx > 0 and fy > 0
    ), "Focal lengths (elements [0, 0] and [1, 1]) must be positive."
    cx, cy = obj[0, 2], obj[1, 2]
    assert (
        cx >= 0 and cy >= 0
    ), "Principal point coordinates (elements [0, 2] and [1, 2]) must be non-negative."
    assert (
        obj[0, 1] == 0
    ), "Camera intrinsics must have zero skew (element [0, 1] must be 0)."
    assert (
        obj[1, 0] == 0
    ), "Camera intrinsics must have zero skew (element [1, 0] must be 0)."
    assert np.array_equal(
        obj[2, :], np.array([0, 0, 1])
    ), "Camera intrinsics must have [0, 0, 1] in the last row."
    return obj


def _check_camera_intrinsics_torch(obj: Any) -> torch.Tensor:
    assert isinstance(obj, torch.Tensor), "Camera intrinsics must be a torch tensor."
    assert obj.ndim == 2, "Camera intrinsics must be a 2D tensor."
    assert obj.shape == (3, 3), "Camera intrinsics must be of shape (3, 3)."
    assert obj.dtype in [
        torch.float32,
        torch.float64,
    ], "Camera intrinsics must be of type float32 or float64."
    fx, fy = obj[0, 0], obj[1, 1]
    assert (
        fx > 0 and fy > 0
    ), "Focal lengths (elements [0, 0] and [1, 1]) must be positive."
    cx, cy = obj[0, 2], obj[1, 2]
    assert (
        cx >= 0 and cy >= 0
    ), "Principal point coordinates (elements [0, 2] and [1, 2]) must be non-negative."
    assert (
        obj[0, 1] == 0
    ), "Camera intrinsics must have zero skew (element [0, 1] must be 0)."
    assert (
        obj[1, 0] == 0
    ), "Camera intrinsics must have zero skew (element [1, 0] must be 0)."
    assert torch.equal(
        obj[2, :], torch.tensor([0, 0, 1], dtype=obj.dtype, device=obj.device)
    ), "Camera intrinsics must have [0, 0, 1] in the last row."
    return obj


def check_camera_intrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, np.ndarray):
        return _check_camera_intrinsics_numpy(obj)
    elif isinstance(obj, torch.Tensor):
        return _check_camera_intrinsics_torch(obj)
    else:
        raise TypeError("Camera intrinsics must be a numpy array or a torch tensor.")


def _check_rotation_matrix_numpy(obj: Any) -> np.ndarray:
    assert isinstance(obj, np.ndarray), "Rotation matrix must be a numpy array."
    assert obj.ndim == 2, "Rotation matrix must be a 2D array."
    assert obj.shape == (3, 3), "Rotation matrix must be of shape (3, 3)."
    assert obj.dtype in [
        np.float32,
        np.float64,
    ], "Rotation matrix must be of type float32 or float64."

    should_be_identity = obj @ obj.T
    max_diff = np.max(np.abs(should_be_identity - np.eye(3)))
    assert np.allclose(
        should_be_identity,
        np.eye(3),
        atol=1.0e-06,
        rtol=0.0,
    ), "Rotation matrix must be orthogonal. Max diff between RR^T and I: {:.6g}".format(max_diff)

    det = np.linalg.det(obj)
    assert np.isclose(
        det,
        1.0,
        atol=1.0e-06,
        rtol=0.0,
    ), f"Rotation matrix must have determinant +1. det(R) = {det:.6g}."

    return obj


def _check_camera_extrinsics_numpy(obj: Any) -> np.ndarray:
    assert isinstance(obj, np.ndarray), "Camera extrinsics must be a numpy array."
    assert obj.ndim == 2, "Camera extrinsics must be a 2D array."
    assert obj.shape == (4, 4), "Camera extrinsics must be of shape (4, 4)."
    assert obj.dtype in [
        np.float32,
        np.float64,
    ], "Camera extrinsics must be of type float32 or float64."
    assert np.allclose(
        obj[3, :], np.array([0, 0, 0, 1])
    ), "Camera extrinsics must have [0, 0, 0, 1] in the last row."
    rotation = obj[:3, :3]
    _check_rotation_matrix_numpy(rotation)
    return obj


def _check_rotation_matrix_torch(obj: Any) -> torch.Tensor:
    assert isinstance(obj, torch.Tensor), "Rotation matrix must be a torch tensor."
    assert obj.ndim == 2, "Rotation matrix must be a 2D tensor."
    assert obj.shape == (3, 3), "Rotation matrix must be of shape (3, 3)."
    assert obj.dtype in [
        torch.float32,
        torch.float64,
    ], "Rotation matrix must be of type float32 or float64."

    identity = torch.eye(3, dtype=obj.dtype, device=obj.device)
    should_be_identity = obj @ obj.T
    max_diff = torch.max(torch.abs(should_be_identity - identity))
    assert torch.allclose(
        should_be_identity,
        identity,
        atol=1.0e-06,
        rtol=0.0,
    ), (
        "Rotation matrix must be orthogonal. Max diff between RR^T and I: "
        f"{float(max_diff)}"
    )

    det = torch.det(obj)
    assert torch.isclose(
        det,
        torch.tensor(1.0, dtype=obj.dtype, device=obj.device),
        atol=1.0e-06,
        rtol=0.0,
    ), f"Rotation matrix must have determinant +1. det(R) = {float(det)}."

    return obj


def _check_camera_extrinsics_torch(obj: Any) -> torch.Tensor:
    assert isinstance(obj, torch.Tensor), "Camera extrinsics must be a torch tensor."
    assert obj.ndim == 2, "Camera extrinsics must be a 2D tensor."
    assert obj.shape == (4, 4), "Camera extrinsics must be of shape (4, 4)."
    assert obj.dtype in [
        torch.float32,
        torch.float64,
    ], "Camera extrinsics must be of type float32 or float64."
    assert torch.equal(
        obj[3, :], torch.tensor([0, 0, 0, 1], dtype=obj.dtype, device=obj.device)
    ), "Camera extrinsics must have [0, 0, 0, 1] in the last row."
    rotation = obj[:3, :3]
    _check_rotation_matrix_torch(rotation)
    return obj


def check_camera_extrinsics(obj: Any) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, np.ndarray):
        return _check_camera_extrinsics_numpy(obj)
    elif isinstance(obj, torch.Tensor):
        return _check_camera_extrinsics_torch(obj)
    else:
        raise TypeError("Camera extrinsics must be a numpy array or a torch tensor.")


def check_rotation_matrix(obj: Any) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(obj, np.ndarray):
        return _check_rotation_matrix_numpy(obj)
    elif isinstance(obj, torch.Tensor):
        return _check_rotation_matrix_torch(obj)
    else:
        raise TypeError("Rotation matrix must be a numpy array or a torch tensor.")
