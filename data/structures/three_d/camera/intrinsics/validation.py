import math
from typing import Any, Dict, Union

import torch


def validate_camera_intrinsics_attributes(
    model: str,
    intr_convention: Any,
    params: Any,
    device: Any,
) -> None:
    """Validate the model, image-plane frame, params, and device for a CameraIntrinsics.

    Single-entry validation for ``CameraIntrinsics.__init__``: validate the camera
    model string, the image-plane convention its params are stated in, those named
    params, and the device together.

    Args:
        model: Camera-model identifier string.
        intr_convention: Candidate image-plane convention the params are stated in.
        params: Candidate named intrinsics params for the model.
        device: Candidate device, expected to be a torch device spec.

    Returns:
        None.
    """
    validate_camera_model(model=model)
    validate_intr_convention(intr_convention=intr_convention)
    validate_camera_intrinsics_params(
        model=model,
        intr_convention=intr_convention,
        params=params,
    )
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


def validate_intr_convention(intr_convention: Any) -> str:
    """Validate an image-plane convention string against the supported set.

    ``standard`` is the pixel raster frame the other three convert through.

    Args:
        intr_convention: Candidate image-plane convention identifier.

    Returns:
        The validated image-plane convention string.
    """
    assert isinstance(intr_convention, str), (
        "Expected camera intrinsics convention to be a string. "
        f"{type(intr_convention)=}"
    )
    assert intr_convention in {"standard", "opengl", "pytorch3d", "vulkan"}, (
        "Expected camera intrinsics convention to be one of "
        "{standard, opengl, pytorch3d, vulkan}. "
        f"{intr_convention=}"
    )
    return intr_convention


def validate_camera_intrinsics_params(
    model: str,
    intr_convention: str,
    params: Any,
) -> Dict[str, Union[int, float]]:
    """Validate the named intrinsics params for a camera model.

    Validates the resolution keys every model carries, the value type every param
    is stated as, the projection keys that model's own dispatch owns, and the
    invariants that hold only across those keys together.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Candidate named intrinsics params for the model.

    Returns:
        The validated named intrinsics params.
    """
    assert isinstance(params, dict), (
        "Expected intrinsics params to be a dict. " f"{type(params)=}"
    )
    for key, value in params.items():
        assert isinstance(value, (int, float)), (
            "Expected every intrinsics param value to be an int or a float. "
            f"{key=} {type(value)=}"
        )
    assert {"h", "w"}.issubset(params.keys()), (
        "Expected intrinsics params to carry the resolution keys h and w. "
        f"{sorted(params.keys())=}"
    )
    assert isinstance(params["h"], int) and isinstance(params["w"], int), (
        "Expected intrinsics params h and w to be ints. "
        f"{type(params['h'])=} {type(params['w'])=}"
    )
    assert params["h"] > 0 and params["w"] > 0, (
        "Expected intrinsics params h and w to be positive. "
        f"{params['h']=} {params['w']=}"
    )

    def _validate_projection_params() -> Dict[str, Union[int, float]]:
        if model == "simple_pinhole":
            return _validate_camera_intrinsics_params_simple_pinhole(params=params)
        if model == "pinhole":
            return _validate_camera_intrinsics_params_pinhole(params=params)
        if model == "ortho":
            return _validate_camera_intrinsics_params_ortho(params=params)
        assert 0, "Should not reach here. " f"{model=}"

    _validate_projection_params()
    validate_camera_intrinsics_invariants(
        model=model,
        intr_convention=intr_convention,
        params=params,
    )
    return params


def _validate_camera_intrinsics_params_simple_pinhole(
    params: Any,
) -> Dict[str, Union[int, float]]:
    """Validate simple_pinhole params: shared focal length f plus principal point.

    Args:
        params: Candidate simple_pinhole params.

    Returns:
        The validated simple_pinhole params.
    """
    assert set(params.keys()) == {"f", "cx", "cy", "h", "w"}, (
        "Expected simple_pinhole params to have exactly keys {f, cx, cy, h, w}. "
        f"{set(params.keys())=}"
    )
    assert params["f"] > 0, (
        "Expected simple_pinhole focal length f to be positive. " f"{params['f']=}"
    )
    assert math.isfinite(float(params["cx"])) and math.isfinite(float(params["cy"])), (
        "Expected simple_pinhole principal point cx / cy to be finite. "
        f"{params['cx']=} {params['cy']=}"
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
    assert set(params.keys()) == {"fx", "fy", "cx", "cy", "h", "w"}, (
        "Expected pinhole params to have exactly keys {fx, fy, cx, cy, h, w}. "
        f"{set(params.keys())=}"
    )
    assert params["fx"] > 0 and params["fy"] > 0, (
        "Expected pinhole focal lengths fx / fy to be positive. "
        f"{params['fx']=} {params['fy']=}"
    )
    assert math.isfinite(float(params["cx"])) and math.isfinite(float(params["cy"])), (
        "Expected pinhole principal point cx / cy to be finite. "
        f"{params['cx']=} {params['cy']=}"
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
    assert set(params.keys()) == {"fx", "fy", "cx", "cy", "h", "w"}, (
        "Expected ortho params to have exactly keys {fx, fy, cx, cy, h, w}. "
        f"{set(params.keys())=}"
    )
    assert params["fx"] > 0 and params["fy"] > 0, (
        "Expected ortho focal scales fx / fy to be positive. "
        f"{params['fx']=} {params['fy']=}"
    )
    assert math.isfinite(float(params["cx"])) and math.isfinite(float(params["cy"])), (
        "Expected ortho principal-point offset cx / cy to be finite. "
        f"{params['cx']=} {params['cy']=}"
    )
    return params


def validate_camera_intrinsics_invariants(
    model: str,
    intr_convention: str,
    params: Dict[str, Union[int, float]],
) -> None:
    """Validate what the intrinsics params state only together.

    The resolution has joined the dict the principal point and the focal already
    live in, and forms a pair with each.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Validated named intrinsics params for the model.

    Returns:
        None.
    """
    _validate_principal_point_within_image(
        model=model,
        intr_convention=intr_convention,
        params=params,
    )
    _validate_model_is_representable_in_frame(
        model=model,
        intr_convention=intr_convention,
        params=params,
    )


def _validate_principal_point_within_image(
    model: str,
    intr_convention: str,
    params: Dict[str, Union[int, float]],
) -> None:
    """Bound a perspective camera's principal point the way its own frame measures it.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Validated named intrinsics params for the model.

    Returns:
        None.
    """
    if model == "ortho":
        # A weak-perspective cx / cy is where the world origin lands rather than where an axis pierces, and a fit drives that off the frame while the camera stays valid.
        return
    cx = float(params["cx"])
    cy = float(params["cy"])
    height = params["h"]
    width = params["w"]
    if intr_convention == "standard":
        assert 0.0 <= cx <= float(width) and 0.0 <= cy <= float(height), (
            "Expected the principal point to fall within the pixel raster running "
            "corner to corner. "
            f"{cx=} {cy=} {height=} {width=}"
        )
        return
    if intr_convention in {"opengl", "vulkan"}:
        assert -1.0 <= cx <= 1.0 and -1.0 <= cy <= 1.0, (
            "Expected the principal point to fall within the device frame, each "
            "axis normalized by its own side. "
            f"{cx=} {cy=} {intr_convention=}"
        )
        return
    if intr_convention == "pytorch3d":
        shorter_side = float(min(height, width))
        assert (
            abs(cx) <= float(width) / shorter_side
            and abs(cy) <= float(height) / shorter_side
        ), (
            "Expected the principal point to fall within the pytorch3d device "
            "frame, whose shorter side alone reaches 1. "
            f"{cx=} {cy=} {height=} {width=}"
        )
        return
    assert 0, "Should not reach here. " f"{intr_convention=}"


def _validate_model_is_representable_in_frame(
    model: str,
    intr_convention: str,
    params: Dict[str, Union[int, float]],
) -> None:
    """Reject a model that states fewer focal params than its frame scales axes.

    A model states as many focal params as it has axes to scale independently, so
    a frame that scales the two axes differently can hold only the models carrying
    two of them.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Validated named intrinsics params for the model.

    Returns:
        None.
    """
    if model == "simple_pinhole" and intr_convention in {"opengl", "vulkan"}:
        assert params["h"] == params["w"], (
            "Expected a square image for a simple_pinhole stated on a device "
            "frame that normalizes each axis by its own side, one shared f "
            "carrying only one unit. "
            f"{params['h']=} {params['w']=} {intr_convention=}"
        )
