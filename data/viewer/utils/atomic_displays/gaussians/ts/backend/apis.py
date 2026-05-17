"""Gaussian-splat display response APIs."""

from typing import Any, Dict, Optional
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.gaussians.ts.backend.core_gaussians_display import (
    create_gaussians_display_response,
)
from data.viewer.utils.atomic_displays.gaussians.ts.backend.schemas.display_response import (
    ColorGSDisplayResponse,
    SegmentationGSDisplayResponse,
)


def create_color_gs_display_response(
    slot_id: str,
    title: str,
    gaussian_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> ColorGSDisplayResponse:
    """Create a color Gaussian-splat display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        gaussian_path: Gaussian-splat artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Color Gaussian-splat display response.
    """
    return create_gaussians_display_response(
        response_type=ColorGSDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="color_gs",
        gaussian_path=gaussian_path,
        meta_info=meta_info,
    )


def create_segmentation_gs_display_response(
    slot_id: str,
    title: str,
    segmentation_gs_path: Optional[str],
    original_overlay_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> SegmentationGSDisplayResponse:
    """Create a segmentation Gaussian-splat display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        segmentation_gs_path: Segmentation Gaussian-splat artifact path.
        original_overlay_path: Original scene artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Segmentation Gaussian-splat display response.
    """
    payload = {} if meta_info is None else dict(meta_info)
    response = create_gaussians_display_response(
        response_type=SegmentationGSDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="segmentation_gs",
        gaussian_path=segmentation_gs_path,
        meta_info=payload,
    )
    if original_overlay_path is not None:
        response.original_overlay_url = "/api/artifacts?%s" % urlencode(
            {"path": original_overlay_path},
        )
    return response
