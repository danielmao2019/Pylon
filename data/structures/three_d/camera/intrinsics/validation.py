import math
from typing import Any, Dict, Union

import torch


def validate_camera_intrinsics_attributes(model: str, params: Any, device: Any) -> None:
    """Validate the model, params, and device for a CameraIntrinsics.

    Single-entry validation for ``CameraIntrinsics.__init__``: validate the camera
    model string, its named params, and the device together.

    Args:
        model: Camera-model identifier string.
        params: Candidate named intrinsics params for the model.
        device: Candidate device, expected to be a torch device spec.

    Returns:
        None.
    """
    validate_camera_model(model=model)
    validate_camera_intrinsics_params(model=model, params=params)
    assert isinstance(device, (str, torch.device)), (
        "Expected CameraIntrinsics device to be a string or torch.device. "
        f"{type(device)=}"
    )


def validate_camera_model(model: Any) -> str:
    """Validate a camera-model string against the supported set.

    Args:
        model: Candidate camera-model identifier.

    Returns:
        The validated camera-model string.
    """
    assert isinstance(model, str), (
        "Expected camera model to be a string. " f"{type(model)=}"
    )
    assert model in {"simple_pinhole", "pinhole", "ortho"}, (
        "Expected camera model to be one of {simple_pinhole, pinhole, ortho}. "
        f"{model=}"
    )
    return model


def validate_camera_intrinsics_params(
    model: str, params: Any
) -> Dict[str, Union[int, float]]:
    """Validate the named intrinsics params for a camera model.

    Dispatches on the model string; all models are structurally equivalent
    siblings, each with its own named-parameter contract.

    Args:
        model: Validated camera-model identifier string.
        params: Candidate named intrinsics params for the model.

    Returns:
        The validated named intrinsics params.
    """
    if model == "simple_pinhole":
        return _validate_camera_intrinsics_params_simple_pinhole(params=params)
    if model == "pinhole":
        return _validate_camera_intrinsics_params_pinhole(params=params)
    if model == "ortho":
        return _validate_camera_intrinsics_params_ortho(params=params)
    assert 0, "Should not reach here. " f"{model=}"


def _validate_camera_intrinsics_params_simple_pinhole(
    params: Any,
) -> Dict[str, Union[int, float]]:
    """Validate simple_pinhole params: shared focal length f plus principal point.

    Args:
        params: Candidate simple_pinhole params.

    Returns:
        The validated simple_pinhole params.
    """
    assert isinstance(params, dict), (
        "Expected simple_pinhole params to be a dict. " f"{type(params)=}"
    )
    assert set(params.keys()) == {"f", "cx", "cy"}, (
        "Expected simple_pinhole params to have exactly keys {f, cx, cy}. "
        f"{set(params.keys())=}"
    )
    assert all(isinstance(value, (int, float)) for value in params.values()), (
        "Expected simple_pinhole params values to be int or float. " f"{params=}"
    )
    assert params["f"] > 0, (
        "Expected simple_pinhole focal length f to be positive. " f"{params['f']=}"
    )
    assert params["cx"] >= 0, (
        "Expected simple_pinhole principal point cx to be non-negative. "
        f"{params['cx']=}"
    )
    assert params["cy"] >= 0, (
        "Expected simple_pinhole principal point cy to be non-negative. "
        f"{params['cy']=}"
    )
    return params


def _validate_camera_intrinsics_params_pinhole(
    params: Any,
) -> Dict[str, Union[int, float]]:
    """Validate pinhole params: independent focal lengths fx / fy plus principal point.

    Args:
        params: Candidate pinhole params.

    Returns:
        The validated pinhole params.
    """
    assert isinstance(params, dict), (
        "Expected pinhole params to be a dict. " f"{type(params)=}"
    )
    assert set(params.keys()) == {"fx", "fy", "cx", "cy"}, (
        "Expected pinhole params to have exactly keys {fx, fy, cx, cy}. "
        f"{set(params.keys())=}"
    )
    assert all(isinstance(value, (int, float)) for value in params.values()), (
        "Expected pinhole params values to be int or float. " f"{params=}"
    )
    assert params["fx"] > 0, (
        "Expected pinhole focal length fx to be positive. " f"{params['fx']=}"
    )
    assert params["fy"] > 0, (
        "Expected pinhole focal length fy to be positive. " f"{params['fy']=}"
    )
    assert params["cx"] >= 0, (
        "Expected pinhole principal point cx to be non-negative. " f"{params['cx']=}"
    )
    assert params["cy"] >= 0, (
        "Expected pinhole principal point cy to be non-negative. " f"{params['cy']=}"
    )
    return params


def _validate_camera_intrinsics_params_ortho(
    params: Any,
) -> Dict[str, Union[int, float]]:
    """Validate ortho (weak-perspective) params: focal scales fx / fy plus offset.

    Args:
        params: Candidate ortho params.

    Returns:
        The validated ortho params.
    """
    assert isinstance(params, dict), (
        "Expected ortho params to be a dict. " f"{type(params)=}"
    )
    assert set(params.keys()) == {"fx", "fy", "cx", "cy"}, (
        "Expected ortho params to have exactly keys {fx, fy, cx, cy}. "
        f"{set(params.keys())=}"
    )
    assert all(isinstance(value, (int, float)) for value in params.values()), (
        "Expected ortho params values to be int or float. " f"{params=}"
    )
    assert params["fx"] > 0, (
        "Expected ortho focal scale fx to be positive. " f"{params['fx']=}"
    )
    assert params["fy"] > 0, (
        "Expected ortho focal scale fy to be positive. " f"{params['fy']=}"
    )
    assert math.isfinite(params["cx"]), (
        "Expected ortho principal-point offset cx to be finite. " f"{params['cx']=}"
    )
    assert math.isfinite(params["cy"]), (
        "Expected ortho principal-point offset cy to be finite. " f"{params['cy']=}"
    )
    return params
