from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch


def rescale_intr_params(
    params: Dict[str, Union[int, float, torch.Tensor]],
    model: str,
    unit_x: Union[int, float, torch.Tensor],
    unit_y: Union[int, float, torch.Tensor],
) -> Dict[str, Union[int, float, torch.Tensor]]:
    """Restate image-plane params under a per-axis unit factor.

    This changes the unit for the focal params and for the cx / cy coordinate or
    offset params; origin translations and axis reversals are handled by the
    convention spokes.

    Args:
        params: The model's named intrinsics params; carries scalar ``cx`` / ``cy`` / ``h`` / ``w`` plus the model's focal key(s) (``f`` for simple_pinhole, ``fx`` / ``fy`` otherwise).
        model: Camera-model identifier string the focal keys belong to.
        unit_x: Horizontal-axis scalar factor every horizontal length is restated by.
        unit_y: Vertical-axis scalar factor every vertical length is restated by.

    Returns:
        A new params dict whose focal and ``cx`` / ``cy`` params are restated in
        the target unit, with ``h`` and ``w`` left where they are.
    """
    params = dict(params)
    params["cx"] = unit_x * params["cx"]
    params["cy"] = unit_y * params["cy"]

    def _rescale_focal(
        params: Dict[str, Union[int, float, torch.Tensor]],
    ) -> Dict[str, Union[int, float, torch.Tensor]]:
        """Scale whichever focal params the model carries.

        Args:
            params: The params dict whose ``cx`` / ``cy`` params are already restated.

        Returns:
            The params dict with its focal params restated in the target unit.
        """
        if model == "simple_pinhole":
            if isinstance(unit_x, torch.Tensor) or isinstance(unit_y, torch.Tensor):
                assert torch.equal(torch.as_tensor(unit_x), torch.as_tensor(unit_y)), (
                    "Expected one shared axis factor for simple_pinhole, whose single "
                    "f cannot carry two different axis scales. "
                    f"{unit_x=} {unit_y=}"
                )
            else:
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
    params: Dict[str, Union[int, float, torch.Tensor]],
    resolution: Optional[
        Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]
    ] = None,
    scale: Optional[
        Union[
            int,
            float,
            Tuple[Union[int, float, torch.Tensor], Union[int, float, torch.Tensor]],
            List[Union[int, float, torch.Tensor]],
            np.ndarray,
            torch.Tensor,
        ]
    ] = None,
) -> Tuple[int, int]:
    """Resolve the two ways a caller names a target resolution into the single form a rescale reads.

    Args:
        params: The model's named intrinsics params, carrying scalar ``h`` / ``w`` values the current resolution is read off.
        resolution: Optional target image resolution as one integer side or ``(height, width)``.
        scale: Optional uniform factor, or a per-axis ``(sx, sy)`` pair, on the resolution the params already carry.

    Returns:
        The target image resolution as an ``(height, width)`` pair of positive ints.
    """

    def _validate_inputs() -> None:
        assert (resolution is None) ^ (scale is None), (
            "Expected exactly one of resolution or scale to be provided. "
            f"{resolution=} {scale=}"
        )
        if resolution is not None:
            assert isinstance(
                resolution, (int, tuple, list, np.ndarray, torch.Tensor)
            ), (
                "Expected resolution to be a positive int or length-2 array-like. "
                f"{type(resolution)=}"
            )
            if isinstance(resolution, int):
                assert resolution > 0, (
                    "Expected scalar resolution to be positive. " f"{resolution=}"
                )
            elif isinstance(resolution, (tuple, list)):
                assert len(resolution) == 2, (
                    "Expected resolution to have length 2. " f"{resolution=}"
                )
                assert all(isinstance(item, int) for item in resolution), (
                    "Expected resolution values to be integers. " f"{resolution=}"
                )
                assert all(item > 0 for item in resolution), (
                    "Expected resolution values to be positive. " f"{resolution=}"
                )
            elif isinstance(resolution, np.ndarray):
                assert resolution.size in (1, 2), (
                    "Expected numpy resolution to contain one or two values. "
                    f"{resolution.shape=}"
                )
                assert np.issubdtype(resolution.dtype, np.integer), (
                    "Expected numpy resolution values to be integers. "
                    f"{resolution.dtype=}"
                )
                assert bool(np.all(resolution > 0)), (
                    "Expected numpy resolution values to be positive. " f"{resolution=}"
                )
            elif isinstance(resolution, torch.Tensor):
                assert resolution.numel() in (1, 2), (
                    "Expected tensor resolution to contain one or two values. "
                    f"{resolution.shape=}"
                )
                assert not resolution.is_floating_point(), (
                    "Expected tensor resolution values to be integers. "
                    f"{resolution.dtype=}"
                )
                assert bool(torch.all(resolution > 0)), (
                    "Expected tensor resolution values to be positive. "
                    f"{resolution=}"
                )
        if scale is not None:
            assert isinstance(
                scale, (int, float, tuple, list, np.ndarray, torch.Tensor)
            ), (
                "Expected scale to be a positive number or length-2 array-like. "
                f"{type(scale)=}"
            )
            if isinstance(scale, (int, float)):
                assert float(scale) > 0.0, (
                    "Expected scalar scale to be positive. " f"{scale=}"
                )
            elif isinstance(scale, (tuple, list)):
                assert len(scale) == 2, "Expected scale to have length 2. " f"{scale=}"
                assert all(
                    isinstance(item, (int, float, torch.Tensor)) for item in scale
                ), (
                    "Expected scale values to be numbers or scalar tensors. "
                    f"{scale=}"
                )
            elif isinstance(scale, np.ndarray):
                assert scale.size in (1, 2), (
                    "Expected numpy scale to contain one or two values. "
                    f"{scale.shape=}"
                )
                assert np.issubdtype(scale.dtype, np.number), (
                    "Expected numpy scale values to be numeric. " f"{scale.dtype=}"
                )
                assert bool(np.all(scale > 0)), (
                    "Expected numpy scale values to be positive. " f"{scale=}"
                )
            elif isinstance(scale, torch.Tensor):
                assert scale.numel() in (1, 2), (
                    "Expected tensor scale to contain one or two values. "
                    f"{scale.shape=}"
                )
                assert scale.is_floating_point(), (
                    "Expected tensor scale values to be floating. " f"{scale.dtype=}"
                )
                assert bool(torch.all(scale > 0)), (
                    "Expected tensor scale values to be positive. " f"{scale=}"
                )

    _validate_inputs()

    def _normalize_inputs(
        resolution: Optional[
            Union[int, Tuple[int, int], List[int], np.ndarray, torch.Tensor]
        ],
        scale: Optional[
            Union[
                int,
                float,
                Tuple[Union[int, float, torch.Tensor], Union[int, float, torch.Tensor]],
                List[Union[int, float, torch.Tensor]],
                np.ndarray,
                torch.Tensor,
            ]
        ],
    ) -> Tuple[
        Optional[Tuple[int, int]],
        Optional[
            Tuple[Union[int, float, torch.Tensor], Union[int, float, torch.Tensor]]
        ],
    ]:
        if resolution is not None:
            if isinstance(resolution, int):
                resolution = (resolution, resolution)
            elif isinstance(resolution, (tuple, list)):
                resolution = (int(resolution[0]), int(resolution[1]))
            elif isinstance(resolution, np.ndarray):
                values = resolution.reshape(-1)
                if values.size == 1:
                    resolution = (int(values[0]), int(values[0]))
                else:
                    resolution = (int(values[0]), int(values[1]))
            elif isinstance(resolution, torch.Tensor):
                values = resolution.reshape(-1)
                if values.numel() == 1:
                    side = int(values[0].detach().cpu().item())
                    resolution = (side, side)
                else:
                    resolution = (
                        int(values[0].detach().cpu().item()),
                        int(values[1].detach().cpu().item()),
                    )
        if scale is not None:
            if isinstance(scale, (int, float)):
                scale = (scale, scale)
            elif isinstance(scale, np.ndarray):
                values = scale.reshape(-1)
                if values.size == 1:
                    scale = (float(values[0]), float(values[0]))
                else:
                    scale = (float(values[0]), float(values[1]))
            elif isinstance(scale, torch.Tensor):
                values = scale.reshape(-1)
                if values.numel() == 1:
                    scale = (values[0], values[0])
                else:
                    scale = (values[0], values[1])
            elif isinstance(scale, list):
                scale = (scale[0], scale[1])
        return resolution, scale

    resolution, scale = _normalize_inputs(resolution=resolution, scale=scale)

    if resolution is not None:
        return resolution
    if scale is not None:
        height = round(
            float(torch.as_tensor(params["h"]).detach().cpu())
            * float(torch.as_tensor(scale[1]).detach().cpu())
        )
        width = round(
            float(torch.as_tensor(params["w"]).detach().cpu())
            * float(torch.as_tensor(scale[0]).detach().cpu())
        )
        assert height > 0 and width > 0, (
            "Expected a scale that keeps both image sides positive. "
            f"{height=} {width=} {scale=}"
        )
        return height, width
    assert 0, "Should not reach here. " f"{resolution=} {scale=}"
