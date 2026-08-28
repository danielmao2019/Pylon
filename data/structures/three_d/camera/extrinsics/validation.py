from typing import Any, List, Union

import numpy as np
import torch

from utils.ops.materialize_tensor import materialize_tensor

_ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS = 32


def validate_camera_extrinsics_attributes(
    extrinsics: Any, extr_convention: Any, device: Any, dtype: Any = None
) -> None:
    """Validate the 4x4 cam2world matrix, pose frame, device, and dtype.

    Single-entry validation for ``CameraExtrinsics.__init__``.

    Args:
        extrinsics: Candidate 4x4 cam2world extrinsics matrix.
        extr_convention: Candidate pose-frame convention string.
        device: Candidate device, expected to be None or a torch device spec.
        dtype: Candidate dtype, expected to be None or a floating torch dtype.

    Returns:
        None.
    """
    validate_extr_convention(extr_convention)
    validate_camera_extrinsics(extrinsics)
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected CameraExtrinsics device to be None, a string, or torch.device. "
        f"{type(device)=}"
    )
    assert dtype is None or isinstance(dtype, torch.dtype), (
        "Expected CameraExtrinsics dtype to be None or a torch dtype. "
        f"{type(dtype)=}"
    )
    if dtype is not None:
        assert torch.empty((), dtype=dtype).is_floating_point(), (
            "Expected CameraExtrinsics dtype to be floating. " f"{dtype=}"
        )


def validate_extr_convention(extr_convention: Any) -> str:
    """Validate a camera-pose convention string against the supported set.

    Args:
        extr_convention: Candidate pose-frame convention string.

    Returns:
        The validated pose-frame convention string.
    """
    assert isinstance(extr_convention, str), f"{type(extr_convention)=}"
    assert extr_convention in [
        "standard",
        "opengl",
        "opencv",
        "pytorch3d",
        "arkit",
    ], f"Unsupported extr_convention: {extr_convention}"
    return extr_convention


def validate_camera_extrinsics(
    obj: Any,
) -> Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]]:
    """Dispatch camera-extrinsics validation on the input representation.

    Args:
        obj: Candidate camera-extrinsics input, a numpy array, torch tensor, or nested numeric list.

    Returns:
        The validated camera-extrinsics array.
    """
    if isinstance(obj, np.ndarray):
        return _validate_camera_extrinsics_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return _validate_camera_extrinsics_torch(obj)
    if isinstance(obj, list):
        return _validate_camera_extrinsics_list(obj)
    raise TypeError(
        "Camera extrinsics must be a numpy array, torch tensor, or nested numeric "
        f"list, got {type(obj)}"
    )


def _validate_camera_extrinsics_numpy(obj: Any) -> np.ndarray:
    """Validate a (..., 4, 4) numpy camera-extrinsics (cam2world) matrix.

    Args:
        obj: Candidate numpy camera-extrinsics array.

    Returns:
        The validated numpy camera-extrinsics array.
    """
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
    """Validate a (..., 4, 4) torch camera-extrinsics (cam2world) matrix.

    Args:
        obj: Candidate torch camera-extrinsics tensor.

    Returns:
        The validated torch camera-extrinsics tensor.
    """
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


def _validate_camera_extrinsics_list(
    obj: List[List[Union[int, float]]],
) -> List[List[Union[int, float]]]:
    """Validate a (4, 4) nested-list camera-extrinsics matrix.

    Args:
        obj: Candidate nested-list camera-to-world matrix.

    Returns:
        The validated nested-list camera-to-world matrix.
    """
    assert isinstance(obj, list), f"{type(obj)=}"
    assert len(obj) == 4, f"{len(obj)=}"
    for row in obj:
        assert isinstance(row, list), (
            "Expected each camera extrinsics row to be a list. " f"{type(row)=}"
        )
        assert len(row) == 4, (
            "Expected each camera extrinsics row to have length 4. " f"{len(row)=}"
        )
        assert all(isinstance(value, (int, float)) for value in row), (
            "Expected each camera extrinsics row to contain numbers. " f"{row=}"
        )
    assert obj[3] == [0, 0, 0, 1], (
        "Camera extrinsics must have [0, 0, 0, 1] in the last row. " f"{obj[3]=}"
    )
    rotation = [row[:3] for row in obj[:3]]
    _validate_rotation_matrix_list(rotation)
    return obj


def validate_rotation_matrix(
    obj: Any,
) -> Union[np.ndarray, torch.Tensor, List[List[Union[int, float]]]]:
    """Dispatch rotation-matrix validation on the input representation.

    Args:
        obj: Candidate rotation-matrix input, a numpy array, torch tensor, or nested numeric list.

    Returns:
        The validated rotation-matrix array.
    """
    if isinstance(obj, np.ndarray):
        return _validate_rotation_matrix_numpy(obj)
    if isinstance(obj, torch.Tensor):
        return _validate_rotation_matrix_torch(obj)
    if isinstance(obj, list):
        return _validate_rotation_matrix_list(obj)
    raise TypeError(
        "Rotation matrix must be a numpy array, torch tensor, or nested numeric "
        f"list, got {type(obj)}"
    )


def _validate_rotation_matrix_numpy(obj: Any) -> np.ndarray:
    """Validate a (..., 3, 3) numpy rotation matrix; dispatch tolerance on dtype.

    Args:
        obj: Candidate numpy rotation-matrix array.

    Returns:
        The validated numpy rotation-matrix array.
    """
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
    """Validate a (..., 3, 3) torch rotation matrix; dispatch tolerance on dtype.

    Args:
        obj: Candidate torch rotation-matrix tensor.

    Returns:
        The validated torch rotation-matrix tensor.
    """
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


def _validate_rotation_matrix_list(
    obj: List[List[Union[int, float]]],
) -> List[List[Union[int, float]]]:
    """Validate a (3, 3) nested-list rotation matrix using float64 tolerance.

    Args:
        obj: Candidate nested-list rotation matrix.

    Returns:
        The validated nested-list rotation matrix.
    """
    assert isinstance(obj, list), f"{type(obj)=}"
    assert len(obj) == 3, f"{len(obj)=}"
    for row in obj:
        assert isinstance(row, list), (
            "Expected each rotation matrix row to be a list. " f"{type(row)=}"
        )
        assert len(row) == 3, (
            "Expected each rotation matrix row to have length 3. " f"{len(row)=}"
        )
        assert all(isinstance(value, (int, float)) for value in row), (
            "Expected each rotation matrix row to contain numbers. " f"{row=}"
        )
    array = np.asarray(obj, dtype=np.float64)
    threshold = _ROTATION_MATRIX_RESIDUAL_FLOOR_ULPS * float(np.finfo(np.float64).eps)
    _validate_rotation_matrix_numpy_against_threshold(obj=array, threshold=threshold)
    return obj


def _validate_rotation_matrix_numpy_against_threshold(
    obj: np.ndarray, threshold: float
) -> np.ndarray:
    """Core numpy rotation check: orthogonality and determinant within atol.

    Args:
        obj: Candidate numpy rotation-matrix array.
        threshold: Absolute tolerance for the orthogonality and determinant checks.

    Returns:
        The validated numpy rotation-matrix array.
    """
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
    """Core torch rotation check: orthogonality and determinant within atol.

    Args:
        obj: Candidate torch rotation-matrix tensor.
        threshold: Absolute tolerance for the orthogonality and determinant checks.

    Returns:
        The validated torch rotation-matrix tensor.
    """
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
