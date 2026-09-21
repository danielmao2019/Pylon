from typing import Any, Dict, Union

import numpy as np
import torch


def validate_camera_intrinsics_attributes(
    model: str,
    intr_convention: Any,
    params: Any,
    device: Any,
    dtype: Any,
) -> None:
    """Validate the model, image-plane frame, params, device, and dtype for a CameraIntrinsics.

    Single-entry validation for ``CameraIntrinsics.__init__``: validate the camera model string, the image-plane convention its params are stated in, those named params, and the optional placement request together.

    Args:
        model: Camera-model identifier string.
        intr_convention: Candidate image-plane convention the params are stated in.
        params: Candidate named intrinsics params for the model, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.
        device: Candidate device, expected to be None or a torch device spec.
        dtype: Candidate dtype, expected to be None or a floating torch dtype.

    Returns:
        None.
    """
    validate_camera_model(model=model)
    validate_intr_convention(intr_convention=intr_convention)
    # The frame goes in ahead of the params, what they mean together depending on it.
    validate_camera_intrinsics_params(
        model=model,
        intr_convention=intr_convention,
        params=params,
    )
    assert device is None or isinstance(device, (str, torch.device)), (
        "Expected CameraIntrinsics device to be None, a string, or torch.device. "
        f"{type(device)=}"
    )
    # An unset device resolves to the one the tensor params share.
    if device is None:
        # The params of one intrinsics are parts of one object, so they sit on one device.
        assert (
            len(
                {
                    value.device
                    for value in params.values()
                    if isinstance(value, torch.Tensor)
                }
            )
            <= 1
        ), (
            "Expected the tensor intrinsics params to share one device when no "
            "device is given. "
            f"{params=}"
        )
    assert dtype is None or isinstance(dtype, torch.dtype), (
        "Expected CameraIntrinsics dtype to be None or a torch dtype. "
        f"{type(dtype)=}"
    )
    if dtype is not None:
        assert torch.empty((), dtype=dtype).is_floating_point(), (
            "Expected CameraIntrinsics dtype to be floating. " f"{dtype=}"
        )
    # An unset dtype resolves to the one the floating params share.
    if dtype is None:
        # The params of one intrinsics are parts of one object, so they hold one dtype.
        assert (
            len(
                {
                    torch.as_tensor(value).dtype
                    for value in params.values()
                    if (isinstance(value, torch.Tensor) and value.is_floating_point())
                    or (
                        isinstance(value, np.ndarray)
                        and np.issubdtype(value.dtype, np.floating)
                    )
                }
            )
            <= 1
        ), (
            "Expected the floating intrinsics params to share one dtype when no "
            "dtype is given. "
            f"{params=}"
        )
    return


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
) -> Dict[str, Union[int, float, np.ndarray, torch.Tensor]]:
    """Validate the named intrinsics params for a camera model.

    Validates the resolution keys every model carries, the projection keys that model's own dispatch owns, and the invariants that hold only across those keys together.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Candidate named intrinsics params for the model, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        The validated named intrinsics params.
    """
    _validate_camera_intrinsics_params_shared(params=params)

    def _validate_projection_params() -> Dict[str, torch.Tensor]:
        """Dispatch the projection keys onto the model that owns them, every model being a structurally equivalent sibling here.

        Args:
            None; reads the enclosing call's ``model`` and ``params``.

        Returns:
            The params, their projection keys validated by the model's own helper.
        """
        if model == "simple_pinhole":
            _validate_camera_intrinsics_params_simple_pinhole(params=params)
            return params
        if model == "pinhole":
            _validate_camera_intrinsics_params_pinhole(params=params)
            return params
        if model == "ortho":
            _validate_camera_intrinsics_params_ortho(params=params)
            return params
        assert 0, "Should not reach here. " f"{model=}"

    _validate_projection_params()
    validate_camera_intrinsics_invariants(
        model=model,
        intr_convention=intr_convention,
        params=params,
    )
    return params


def _validate_camera_intrinsics_params_shared(
    params: Any,
) -> Dict[str, Union[int, float, np.ndarray, torch.Tensor]]:
    """Validate what the params of every model share: numbers or arrays of at most one axis and one shape, carrying a positive resolution h and w.

    Args:
        params: Candidate named intrinsics params, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        The validated named intrinsics params.
    """
    # Checked first, since every check below reads a value's shape.
    assert isinstance(params, dict), (
        "Expected intrinsics params to be a dict. " f"{type(params)=}"
    )
    for key, value in params.items():
        assert isinstance(key, str), (
            "Expected every intrinsics params key to be a string. "
            f"{key=} {type(key)=}"
        )
        assert isinstance(value, (int, float, np.ndarray, torch.Tensor)), (
            "Expected every intrinsics param value to be an int, a float, a numpy "
            f"array, or a torch.Tensor. {key=} {type(value)=}"
        )
        if isinstance(value, (int, float)):
            continue
        # The normalization casts every param onto one floating dtype, which would turn a bool into 0 / 1 without a word.
        if isinstance(value, np.ndarray):
            assert np.issubdtype(value.dtype, np.number), (
                "Expected every numpy intrinsics param to be numeric. "
                f"{key=} {value.dtype=}"
            )
            assert value.ndim <= 1, (
                "Expected every numpy intrinsics param to be a scalar or a one-axis "
                f"batch. {key=} {value.shape=}"
            )
            continue
        # That same cast would drop an imaginary part without a word.
        if isinstance(value, torch.Tensor):
            assert not value.is_complex(), (
                "Expected every tensor intrinsics param to be real-valued. "
                f"{key=} {value.dtype=}"
            )
            assert value.ndim <= 1, (
                "Expected every tensor intrinsics param to be a scalar or a one-axis "
                f"batch. {key=} {value.shape=}"
            )
            continue
        assert 0, "Should not reach here."
    batch_shapes = {}
    for key, value in params.items():
        batch_shapes[key] = tuple(np.shape(value))
    assert len(set(batch_shapes.values())) == 1, (
        "Expected every intrinsics param to share one leading batch shape, a scalar "
        f"param being the empty-batch case. {batch_shapes=}"
    )
    # The resolution, named the way every resolution in this repo is ordered: h first.
    assert {"h", "w"}.issubset(params.keys()), (
        "Expected intrinsics params to carry the resolution keys h and w. "
        f"{sorted(params.keys())=}"
    )
    assert bool(torch.all(torch.as_tensor(params["h"]) > 0)) and bool(
        torch.all(torch.as_tensor(params["w"]) > 0)
    ), (
        "Expected intrinsics params h and w to be positive at every entry. "
        f"{params['h']=} {params['w']=}"
    )
    return params


def _validate_camera_intrinsics_params_simple_pinhole(
    params: Any,
) -> Dict[str, Union[int, float, np.ndarray, torch.Tensor]]:
    """Validate simple_pinhole params: shared focal length f plus principal point.

    Args:
        params: Candidate simple_pinhole params, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        The validated simple_pinhole params.
    """
    assert set(params.keys()) == {"f", "cx", "cy", "h", "w"}, (
        "Expected simple_pinhole params to have exactly keys {f, cx, cy, h, w}. "
        f"{set(params.keys())=}"
    )
    assert bool(torch.all(torch.as_tensor(params["f"]) > 0)), (
        "Expected simple_pinhole focal length f to be positive. " f"{params['f']=}"
    )
    assert bool(torch.all(torch.isfinite(torch.as_tensor(params["cx"])))) and bool(
        torch.all(torch.isfinite(torch.as_tensor(params["cy"])))
    ), (
        "Expected simple_pinhole principal point cx / cy to be finite at every "
        "entry. "
        f"{params['cx']=} {params['cy']=}"
    )
    return params


def _validate_camera_intrinsics_params_pinhole(
    params: Any,
) -> Dict[str, Union[int, float, np.ndarray, torch.Tensor]]:
    """Validate pinhole params: independent focal lengths fx / fy plus principal point.

    Args:
        params: Candidate pinhole params, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        The validated pinhole params.
    """
    assert set(params.keys()) == {"fx", "fy", "cx", "cy", "h", "w"}, (
        "Expected pinhole params to have exactly keys {fx, fy, cx, cy, h, w}. "
        f"{set(params.keys())=}"
    )
    assert bool(torch.all(torch.as_tensor(params["fx"]) > 0)) and bool(
        torch.all(torch.as_tensor(params["fy"]) > 0)
    ), (
        "Expected pinhole focal lengths fx / fy to be positive. "
        f"{params['fx']=} {params['fy']=}"
    )
    assert bool(torch.all(torch.isfinite(torch.as_tensor(params["cx"])))) and bool(
        torch.all(torch.isfinite(torch.as_tensor(params["cy"])))
    ), (
        "Expected pinhole principal point cx / cy to be finite. "
        f"{params['cx']=} {params['cy']=}"
    )
    return params


def _validate_camera_intrinsics_params_ortho(
    params: Any,
) -> Dict[str, Union[int, float, np.ndarray, torch.Tensor]]:
    """Validate ortho (weak-perspective) params: focal scales fx / fy plus offset.

    Args:
        params: Candidate ortho params, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        The validated ortho params.
    """
    assert set(params.keys()) == {"fx", "fy", "cx", "cy", "h", "w"}, (
        "Expected ortho params to have exactly keys {fx, fy, cx, cy, h, w}. "
        f"{set(params.keys())=}"
    )
    assert bool(torch.all(torch.as_tensor(params["fx"]) > 0)) and bool(
        torch.all(torch.as_tensor(params["fy"]) > 0)
    ), (
        "Expected ortho focal scales fx / fy to be positive. "
        f"{params['fx']=} {params['fy']=}"
    )
    assert bool(torch.all(torch.isfinite(torch.as_tensor(params["cx"])))) and bool(
        torch.all(torch.isfinite(torch.as_tensor(params["cy"])))
    ), (
        "Expected ortho principal-point offset cx / cy to be finite. "
        f"{params['cx']=} {params['cy']=}"
    )
    return params


def validate_camera_intrinsics_invariants(
    model: str,
    intr_convention: str,
    params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
) -> None:
    """Validate what the intrinsics params state only together.

    The resolution has joined the dict the principal point and the focal already live in, and forms a pair with each.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Validated named intrinsics params for the model, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

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
    return


def _validate_principal_point_within_image(
    model: str,
    intr_convention: str,
    params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
) -> None:
    """Bound a perspective camera's principal point the way its own frame measures it.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Validated named intrinsics params for the model, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        None.
    """
    if model == "ortho":
        # A weak-perspective cx / cy is where the world origin lands rather than where an axis pierces, and a fit drives that off the frame while the camera stays valid.
        return
    if intr_convention == "standard":
        # The pixel frame running corner to corner.
        assert bool(
            torch.all(
                (torch.as_tensor(params["cx"]) >= 0.0)
                & (torch.as_tensor(params["cx"]) <= torch.as_tensor(params["w"]))
            )
        ) and bool(
            torch.all(
                (torch.as_tensor(params["cy"]) >= 0.0)
                & (torch.as_tensor(params["cy"]) <= torch.as_tensor(params["h"]))
            )
        ), (
            "Expected the principal point to fall within the pixel raster running "
            "corner to corner. "
            f"{params['cx']=} {params['cy']=} {params['h']=} {params['w']=}"
        )
        return
    if intr_convention in {"opengl", "vulkan"}:
        # Each axis normalized by its own side, so both bounds are the same.
        assert bool(
            torch.all(torch.abs(torch.as_tensor(params["cx"])) <= 1.0)
        ) and bool(torch.all(torch.abs(torch.as_tensor(params["cy"])) <= 1.0)), (
            "Expected the principal point to fall within the device frame, each "
            "axis normalized by its own side. "
            f"{params['cx']=} {params['cy']=} {intr_convention=}"
        )
        return
    if intr_convention == "pytorch3d":
        # The shorter side alone reaches 1, so the longer axis's bound is the larger.
        assert bool(
            torch.all(
                torch.abs(torch.as_tensor(params["cx"]))
                <= torch.as_tensor(params["w"])
                / torch.minimum(
                    torch.as_tensor(params["h"]), torch.as_tensor(params["w"])
                )
            )
        ) and bool(
            torch.all(
                torch.abs(torch.as_tensor(params["cy"]))
                <= torch.as_tensor(params["h"])
                / torch.minimum(
                    torch.as_tensor(params["h"]), torch.as_tensor(params["w"])
                )
            )
        ), (
            "Expected the principal point to fall within the pytorch3d device "
            "frame, whose shorter side alone reaches 1. "
            f"{params['cx']=} {params['cy']=} {params['h']=} {params['w']=}"
        )
        return
    assert 0, "Should not reach here. " f"{intr_convention=}"


def _validate_model_is_representable_in_frame(
    model: str,
    intr_convention: str,
    params: Dict[str, Union[int, float, np.ndarray, torch.Tensor]],
) -> None:
    """Reject a model that states fewer focal params than its frame scales axes.

    A model states as many focal params as it has axes to scale independently, so a frame that scales the two axes differently can hold only the models carrying two of them.

    Args:
        model: Validated camera-model identifier string.
        intr_convention: Validated image-plane convention the params are stated in.
        params: Validated named intrinsics params for the model, each an int, a float, or a ``[]`` / ``[B]`` numpy array or torch.Tensor.

    Returns:
        None.
    """
    if model == "simple_pinhole" and intr_convention in {"opengl", "vulkan"}:
        # These frames normalize each axis by its own side, and one shared f cannot carry two different units, so a non-square image has no simple_pinhole in them.
        assert bool(
            torch.all(torch.as_tensor(params["h"]) == torch.as_tensor(params["w"]))
        ), (
            "Expected a square image for a simple_pinhole stated on a device "
            "frame that normalizes each axis by its own side, one shared f "
            "carrying only one unit. "
            f"{params['h']=} {params['w']=} {intr_convention=}"
        )
    return
