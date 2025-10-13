from typing import Optional, Tuple, Union

import torch

from utils.input_checks.check_camera import check_camera_intrinsics


def scale_intrinsics(
    intrinsics: torch.Tensor,
    resolution: Optional[Tuple[int, int]] = None,
    scale: Optional[
        Union[Union[int, float], Tuple[Union[int, float], Union[int, float]]]
    ] = None,
    inplace: bool = False,
):
    """Scale a pinhole camera intrinsics matrix for a new resolution or by a scalar.

    Args:
        intrinsics: A 3x3 camera intrinsics matrix as a torch.Tensor.
        resolution: Optional target image resolution as (height, width).
        scale: Optional uniform scale factor applied to both axes.

    Returns:
        torch.Tensor: Scaled intrinsics (3x3), same dtype/device as input.
    """

    # ---- Input validation ----
    # 1) Assert-check camera intrinsics using provided API
    check_camera_intrinsics(intrinsics)

    # 1.a) Inplace must be a boolean
    assert isinstance(inplace, bool), "'inplace' must be a boolean."

    # 2) Exactly one of resolution or scale must be provided (XOR)
    if not ((resolution is not None) ^ (scale is not None)):
        raise AssertionError(
            "Provide exactly one of 'resolution' or 'scale' (but not both)."
        )

    # 3) If scale is None, validate resolution (image resolution: H, W) and define per-axis scale tuple now
    if scale is None:
        assert (
            resolution is not None
        ), "When 'scale' is None, 'resolution' must be provided."
        assert (
            isinstance(resolution, tuple) and len(resolution) == 2
        ), "'resolution' must be a tuple of length 2: (height, width)."
        assert isinstance(resolution[0], int) and isinstance(
            resolution[1], int
        ), "'resolution' values must be integers: (height:int, width:int)."
        assert (
            resolution[0] > 0 and resolution[1] > 0
        ), "'resolution' values must be positive integers."
        # Compute per-axis scale directly here by inferring original size from intrinsics
        original_width = int(intrinsics[0, 2].item() * 2)
        original_height = int(intrinsics[1, 2].item() * 2)
        if original_width <= 0 or original_height <= 0:
            raise ValueError(
                "Cannot infer original size from intrinsics principal point; got non-positive width/height."
            )
        # resolution is (H, W) so map to per-axis (sx, sy) = (W_scale, H_scale)
        target_height, target_width = resolution
        scale = (
            float(target_width) / float(original_width),
            float(target_height) / float(original_height),
        )

    # 4) Validate 'scale' form directly (scalar or pair) when provided
    assert isinstance(scale, (int, float)) or (
        isinstance(scale, tuple)
        and len(scale) == 2
        and isinstance(scale[0], (int, float))
        and isinstance(scale[1], (int, float))
    ), "'scale' must be a positive number or a tuple (sx, sy) of positive numbers."
    if isinstance(scale, (int, float)):
        assert float(scale) > 0, "'scale' must be a positive number."
    if isinstance(scale, tuple):
        assert (
            float(scale[0]) > 0 and float(scale[1]) > 0
        ), "'scale' tuple values must be positive numbers."

    # ---- Scaling logic ----
    # Clone before modification if not inplace
    if not inplace:
        intrinsics = intrinsics.clone()

    # Apply scale to intrinsics
    intrinsics[0, 0] = intrinsics[0, 0] * float(scale[0])  # fx
    intrinsics[1, 1] = intrinsics[1, 1] * float(scale[1])  # fy
    intrinsics[0, 2] = intrinsics[0, 2] * float(scale[0])  # cx
    intrinsics[1, 2] = intrinsics[1, 2] * float(scale[1])  # cy

    # Final check on resulting intrinsics
    check_camera_intrinsics(intrinsics)

    return intrinsics
