from typing import Dict, Optional, Tuple, Union


def scale_camera_intrinsics_params(
    params: Dict[str, Union[int, float]],
    resolution: Optional[Tuple[int, int]] = None,
    scale: Optional[
        Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
    ] = None,
) -> Dict[str, Union[int, float]]:
    """Scale a camera model's intrinsics params to a resolution or by a factor.

    Resolves a per-axis ``(sx, sy)`` scale either from a target resolution (target
    over the current resolution inferred from the principal point) or directly from
    a scale factor, then scales the focal params (``f`` / ``fx`` / ``fy``) and the
    principal-point params (``cx`` / ``cy``) in a copy of the params.

    Args:
        params: The model's named intrinsics params; carries ``cx`` / ``cy`` plus
            the model's focal key(s) (``f`` for simple_pinhole, ``fx`` / ``fy``
            otherwise).
        resolution: Optional target image resolution as ``(height, width)``.
        scale: Optional uniform scale, or a per-axis ``(sx, sy)`` tuple.

    Returns:
        A new params dict with the focal and principal-point params scaled by the
        resolved ``(sx, sy)``.
    """
    # Input validations
    assert isinstance(params, dict), (
        "Expected intrinsics params to be a dict. " f"{type(params)=}"
    )
    assert (resolution is None) ^ (scale is None), (
        "Expected exactly one of resolution or scale to be provided. "
        f"{resolution=} {scale=}"
    )

    # Input normalizations
    if scale is None:
        assert isinstance(resolution, tuple) and len(resolution) == 2, (
            "Expected resolution to be a (height, width) tuple of length 2. "
            f"{resolution=}"
        )
        assert isinstance(resolution[0], int) and isinstance(resolution[1], int), (
            "Expected resolution values to be integers (height, width). "
            f"{resolution=}"
        )
        assert resolution[0] > 0 and resolution[1] > 0, (
            "Expected resolution values to be positive integers. " f"{resolution=}"
        )
        current_width = float(params["cx"]) * 2.0
        current_height = float(params["cy"]) * 2.0
        assert current_width > 0.0 and current_height > 0.0, (
            "Expected positive current resolution inferred from the principal point. "
            f"{current_width=} {current_height=}"
        )
        target_height, target_width = resolution
        scale_x = float(target_width) / current_width
        scale_y = float(target_height) / current_height
    elif isinstance(scale, (int, float)):
        assert float(scale) > 0.0, "Expected scalar scale to be positive. " f"{scale=}"
        scale_x = float(scale)
        scale_y = float(scale)
    else:
        assert isinstance(scale, tuple) and len(scale) == 2, (
            "Expected scale to be a number or an (sx, sy) tuple of length 2. "
            f"{scale=}"
        )
        assert isinstance(scale[0], (int, float)) and isinstance(
            scale[1], (int, float)
        ), ("Expected scale tuple values to be numbers. " f"{scale=}")
        assert float(scale[0]) > 0.0 and float(scale[1]) > 0.0, (
            "Expected scale tuple values to be positive. " f"{scale=}"
        )
        scale_x = float(scale[0])
        scale_y = float(scale[1])

    # Scaling logic
    scaled_params: Dict[str, Union[int, float]] = dict(params)
    if "f" in scaled_params:
        scaled_params["f"] = scaled_params["f"] * scale_x
    if "fx" in scaled_params:
        scaled_params["fx"] = scaled_params["fx"] * scale_x
    if "fy" in scaled_params:
        scaled_params["fy"] = scaled_params["fy"] * scale_y
    scaled_params["cx"] = scaled_params["cx"] * scale_x
    scaled_params["cy"] = scaled_params["cy"] * scale_y
    return scaled_params
