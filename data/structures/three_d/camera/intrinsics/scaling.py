from typing import Dict, Optional, Tuple, Union


def rescale_intr_params(
    params: Dict[str, Union[int, float]],
    model: str,
    unit_x: float,
    unit_y: float,
) -> Dict[str, Union[int, float]]:
    """Restate image-plane params under a per-axis unit factor.

    This changes the unit for the focal params and for the cx / cy coordinate or
    offset params; origin translations and axis reversals are handled by the
    convention spokes.

    Args:
        params: The model's named intrinsics params; carries ``cx`` / ``cy`` / ``h`` / ``w`` plus the model's focal key(s) (``f`` for simple_pinhole, ``fx`` / ``fy`` otherwise).
        model: Camera-model identifier string the focal keys belong to.
        unit_x: Horizontal-axis factor every horizontal length is restated by.
        unit_y: Vertical-axis factor every vertical length is restated by.

    Returns:
        A new params dict whose focal and ``cx`` / ``cy`` params are restated in
        the target unit, with ``h`` and ``w`` left where they are.
    """
    params = dict(params)
    params["cx"] = unit_x * params["cx"]
    params["cy"] = unit_y * params["cy"]

    def _rescale_focal(
        params: Dict[str, Union[int, float]],
    ) -> Dict[str, Union[int, float]]:
        """Scale whichever focal params the model carries.

        Args:
            params: The params dict whose ``cx`` / ``cy`` params are already restated.

        Returns:
            The params dict with its focal params restated in the target unit.
        """
        if model == "simple_pinhole":
            assert unit_x == unit_y, (
                "Expected one shared axis factor for simple_pinhole, whose single "
                "f cannot carry two different axis scales. "
                f"{unit_x=} {unit_y=}"
            )
            params["f"] = unit_x * params["f"]
            return params
        if model in {"pinhole", "ortho"}:
            params["fx"] = unit_x * params["fx"]
            params["fy"] = unit_y * params["fy"]
            return params
        raise NotImplementedError(
            "No focal rescale rule for this camera model. " f"{model=}"
        )

    params = _rescale_focal(params=params)
    return params


def resolve_target_resolution(
    params: Dict[str, Union[int, float]],
    resolution: Optional[Tuple[int, int]] = None,
    scale: Optional[
        Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
    ] = None,
) -> Tuple[int, int]:
    """Resolve the two ways a caller names a target resolution into the single form a rescale reads.

    Args:
        params: The model's named intrinsics params, carrying the ``h`` / ``w`` the current resolution is read off.
        resolution: Optional target image resolution as ``(height, width)``.
        scale: Optional uniform factor, or a per-axis ``(sx, sy)`` tuple, on the resolution the params already carry.

    Returns:
        The target image resolution as an ``(height, width)`` pair of positive ints.
    """

    def _validate_inputs() -> None:
        assert (resolution is None) ^ (scale is None), (
            "Expected exactly one of resolution or scale to be provided. "
            f"{resolution=} {scale=}"
        )
        if resolution is not None:
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
        if scale is not None:
            if isinstance(scale, (int, float)):
                assert float(scale) > 0.0, (
                    "Expected scalar scale to be positive. " f"{scale=}"
                )
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

    _validate_inputs()

    def _normalize_inputs() -> Optional[Tuple[Union[int, float], Union[int, float]]]:
        if isinstance(scale, (int, float)):
            return scale, scale
        if scale is None or isinstance(scale, tuple):
            return scale
        assert 0, "Should not reach here. " f"{scale=}"

    scale = _normalize_inputs()

    if resolution is not None:
        return resolution
    if scale is not None:
        height = round(float(params["h"]) * float(scale[1]))
        width = round(float(params["w"]) * float(scale[0]))
        assert height > 0 and width > 0, (
            "Expected a scale that keeps both image sides positive. "
            f"{height=} {width=} {scale=}"
        )
        return height, width
    assert 0, "Should not reach here. " f"{resolution=} {scale=}"
