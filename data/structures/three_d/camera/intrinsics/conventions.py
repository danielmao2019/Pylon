from typing import Dict, Tuple, Union

from data.structures.three_d.camera.intrinsics.scaling import rescale_intr_params


def transform_intr_convention(
    params: Dict[str, Union[int, float]],
    model: str,
    source_intr_convention: str,
    target_intr_convention: str,
) -> Dict[str, Union[int, float]]:
    """Restate one camera model's named params from one image-plane frame into another.

    Routed through the standard frame, so each frame brings its own two helpers
    rather than one against every frame already here.

    Args:
        params: The model's named intrinsics params stated on ``source_intr_convention``; carries ``cx`` / ``cy`` / ``h`` / ``w`` plus the model's focal key(s).
        model: Camera-model identifier string the focal keys belong to.
        source_intr_convention: Image-plane frame the params are stated in.
        target_intr_convention: Image-plane frame the params are restated on.

    Returns:
        The params restated on ``target_intr_convention``.
    """
    if source_intr_convention == target_intr_convention:
        return params

    def _to_standard(
        params: Dict[str, Union[int, float]],
    ) -> Dict[str, Union[int, float]]:
        """Dispatch the source frame onto its own inbound spoke.

        Args:
            params: The params as the caller stated them.

        Returns:
            The params on the standard pixel frame.
        """
        if source_intr_convention == "standard":
            return params
        if source_intr_convention == "opengl":
            return _opengl_to_standard(params=params, model=model)
        if source_intr_convention == "pytorch3d":
            return _pytorch3d_to_standard(params=params, model=model)
        if source_intr_convention == "vulkan":
            return _vulkan_to_standard(params=params, model=model)
        assert 0, "Should not reach here. " f"{source_intr_convention=}"

    params = _to_standard(params=params)

    def _from_standard(
        params: Dict[str, Union[int, float]],
    ) -> Dict[str, Union[int, float]]:
        """Dispatch the target frame onto its own outbound spoke.

        Args:
            params: The params on the standard pixel frame.

        Returns:
            The params on the target frame.
        """
        if target_intr_convention == "standard":
            return params
        if target_intr_convention == "opengl":
            return _standard_to_opengl(params=params, model=model)
        if target_intr_convention == "pytorch3d":
            return _standard_to_pytorch3d(params=params, model=model)
        if target_intr_convention == "vulkan":
            return _standard_to_vulkan(params=params, model=model)
        assert 0, "Should not reach here. " f"{target_intr_convention=}"

    params = _from_standard(params=params)
    return params


def _standard_to_opengl(
    params: Dict[str, Union[int, float]],
    model: str,
) -> Dict[str, Union[int, float]]:
    """Restate pixel params on OpenGL's device frame.

    Its origin is the image's centre, its x runs with standard's toward the right
    edge and its y against it toward the top, each axis spanning its own side.

    Args:
        params: The model's named intrinsics params on the standard pixel frame.
        model: Camera-model identifier string the focal keys belong to.

    Returns:
        The params on the opengl frame.
    """
    unit_x = 2.0 / float(params["w"])
    unit_y = 2.0 / float(params["h"])
    params = _centre_principal_point(params=params)
    params = _reverse_axes(params=params, axes=("y",))
    params = rescale_intr_params(
        params=params,
        model=model,
        unit_x=unit_x,
        unit_y=unit_y,
    )
    return params


def _opengl_to_standard(
    params: Dict[str, Union[int, float]],
    model: str,
) -> Dict[str, Union[int, float]]:
    """Restate OpenGL device params back on the standard pixel frame.

    The inbound half of the same frame, the three steps run in reverse so a round
    trip returns what it started as.

    Args:
        params: The model's named intrinsics params on the opengl frame.
        model: Camera-model identifier string the focal keys belong to.

    Returns:
        The params on the standard frame.
    """
    unit_x = float(params["w"]) / 2.0
    unit_y = float(params["h"]) / 2.0
    params = rescale_intr_params(
        params=params,
        model=model,
        unit_x=unit_x,
        unit_y=unit_y,
    )
    params = _reverse_axes(params=params, axes=("y",))
    params = _uncentre_principal_point(params=params)
    return params


def _standard_to_pytorch3d(
    params: Dict[str, Union[int, float]],
    model: str,
) -> Dict[str, Union[int, float]]:
    """Restate pixel params on PyTorch3D's device frame.

    Its origin is the image's centre, its x runs toward the left edge and its y
    toward the top, and its shorter side alone spans ``[-1, 1]``.

    Args:
        params: The model's named intrinsics params on the standard pixel frame.
        model: Camera-model identifier string the focal keys belong to.

    Returns:
        The params on the pytorch3d frame.
    """
    unit = 2.0 / float(min(params["h"], params["w"]))
    params = _centre_principal_point(params=params)
    params = _reverse_axes(params=params, axes=("x", "y"))
    params = rescale_intr_params(
        params=params,
        model=model,
        unit_x=unit,
        unit_y=unit,
    )
    return params


def _pytorch3d_to_standard(
    params: Dict[str, Union[int, float]],
    model: str,
) -> Dict[str, Union[int, float]]:
    """Restate PyTorch3D device params back on the standard pixel frame.

    Args:
        params: The model's named intrinsics params on the pytorch3d frame.
        model: Camera-model identifier string the focal keys belong to.

    Returns:
        The params on the standard frame.
    """
    unit = float(min(params["h"], params["w"])) / 2.0
    params = rescale_intr_params(
        params=params,
        model=model,
        unit_x=unit,
        unit_y=unit,
    )
    params = _reverse_axes(params=params, axes=("x", "y"))
    params = _uncentre_principal_point(params=params)
    return params


def _standard_to_vulkan(
    params: Dict[str, Union[int, float]],
    model: str,
) -> Dict[str, Union[int, float]]:
    """Restate pixel params on Vulkan's device frame.

    It agrees with standard on both axis directions, and differs from OpenGL's in
    exactly that.

    Args:
        params: The model's named intrinsics params on the standard pixel frame.
        model: Camera-model identifier string the focal keys belong to.

    Returns:
        The params on the vulkan frame.
    """
    unit_x = 2.0 / float(params["w"])
    unit_y = 2.0 / float(params["h"])
    params = _centre_principal_point(params=params)
    params = rescale_intr_params(
        params=params,
        model=model,
        unit_x=unit_x,
        unit_y=unit_y,
    )
    return params


def _vulkan_to_standard(
    params: Dict[str, Union[int, float]],
    model: str,
) -> Dict[str, Union[int, float]]:
    """Restate Vulkan device params back on the standard pixel frame.

    Args:
        params: The model's named intrinsics params on the vulkan frame.
        model: Camera-model identifier string the focal keys belong to.

    Returns:
        The params on the standard frame.
    """
    unit_x = float(params["w"]) / 2.0
    unit_y = float(params["h"]) / 2.0
    params = rescale_intr_params(
        params=params,
        model=model,
        unit_x=unit_x,
        unit_y=unit_y,
    )
    params = _uncentre_principal_point(params=params)
    return params


def _centre_principal_point(
    params: Dict[str, Union[int, float]],
) -> Dict[str, Union[int, float]]:
    """Move the principal point off the image's top-left corner onto its centre.

    Args:
        params: The model's named intrinsics params on a corner origin.

    Returns:
        The params on a centred origin.
    """
    params = dict(params)
    params["cx"] = params["cx"] - float(params["w"]) / 2.0
    params["cy"] = params["cy"] - float(params["h"]) / 2.0
    return params


def _uncentre_principal_point(
    params: Dict[str, Union[int, float]],
) -> Dict[str, Union[int, float]]:
    """Move the principal point back off the image's centre onto its top-left corner.

    Args:
        params: The model's named intrinsics params on a centred origin.

    Returns:
        The params on a corner origin.
    """
    params = dict(params)
    params["cx"] = params["cx"] + float(params["w"]) / 2.0
    params["cy"] = params["cy"] + float(params["h"]) / 2.0
    return params


def _reverse_axes(
    params: Dict[str, Union[int, float]],
    axes: Tuple[str, ...],
) -> Dict[str, Union[int, float]]:
    """Reverse the named image axes, which reaches the principal point alone.

    The focal params are left as they are: the reversal reaches the linear term at
    both ends, the camera-space coordinate feeding it and the image coordinate it
    produces, and those two cancel.

    Args:
        params: The model's named intrinsics params.
        axes: The image axes to reverse, each ``"x"`` or ``"y"``.

    Returns:
        The params on the reversed axes.
    """
    params = dict(params)
    for axis in axes:
        assert axis in {"x", "y"}, (
            "Expected each reversed image axis to be x or y. " f"{axis=} {axes=}"
        )
        params[f"c{axis}"] = -params[f"c{axis}"]
    return params
