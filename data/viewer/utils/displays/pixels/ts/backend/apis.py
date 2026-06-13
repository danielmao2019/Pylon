"""Pixel display response APIs."""

from typing import Any, Dict, Optional
from urllib.parse import urlencode

from data.viewer.utils.displays.pixels.ts.backend.core_pixels_display import (
    create_pixels_display_response_core,
)
from data.viewer.utils.displays.pixels.ts.backend.schemas.display_response import (
    ColorImageDisplayResponse,
    DepthImageDisplayResponse,
    EdgeImageDisplayResponse,
    InstanceSurrogateImageDisplayResponse,
    NormalImageDisplayResponse,
    SegmentationImageDisplayResponse,
)


def create_color_image_display_response(
    slot_id: str,
    title: str,
    image_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> ColorImageDisplayResponse:
    """Create a color image display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        image_path: Image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Color image display response.
    """
    return create_pixels_display_response_core(
        response_type=ColorImageDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="color_image",
        image_path=image_path,
        meta_info=meta_info,
    )


def create_depth_image_display_response(
    slot_id: str,
    title: str,
    image_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> DepthImageDisplayResponse:
    """Create a depth image display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        image_path: Image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Depth image display response.
    """
    return create_pixels_display_response_core(
        response_type=DepthImageDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="depth_image",
        image_path=image_path,
        meta_info=meta_info,
    )


def create_edge_image_display_response(
    slot_id: str,
    title: str,
    image_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> EdgeImageDisplayResponse:
    """Create an edge image display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        image_path: Image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Edge image display response.
    """
    return create_pixels_display_response_core(
        response_type=EdgeImageDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="edge_image",
        image_path=image_path,
        meta_info=meta_info,
    )


def create_normal_image_display_response(
    slot_id: str,
    title: str,
    image_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> NormalImageDisplayResponse:
    """Create a normal image display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        image_path: Image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Normal image display response.
    """
    return create_pixels_display_response_core(
        response_type=NormalImageDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="normal_image",
        image_path=image_path,
        meta_info=meta_info,
    )


def create_segmentation_image_display_response(
    slot_id: str,
    title: str,
    segmentation_image_path: Optional[str],
    original_overlay_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> SegmentationImageDisplayResponse:
    """Create a segmentation image display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        segmentation_image_path: Segmentation image artifact path.
        original_overlay_path: Original image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Segmentation image display response.
    """
    payload = {} if meta_info is None else dict(meta_info)
    response = create_pixels_display_response_core(
        response_type=SegmentationImageDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="segmentation_image",
        image_path=segmentation_image_path,
        meta_info=payload,
    )
    if original_overlay_path is not None:
        response.original_overlay_url = "/api/artifacts?%s" % urlencode(
            {"path": original_overlay_path},
        )
    return response


def create_instance_surrogate_image_display_response(
    slot_id: str,
    title: str,
    image_path: Optional[str],
    original_overlay_path: Optional[str] = None,
    meta_info: Dict[str, Any] | None = None,
) -> InstanceSurrogateImageDisplayResponse:
    """Create an instance-surrogate image display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        image_path: Image artifact path.
        original_overlay_path: Original image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Instance-surrogate image display response.
    """
    response = create_pixels_display_response_core(
        response_type=InstanceSurrogateImageDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="instance_surrogate_image",
        image_path=image_path,
        meta_info=meta_info,
    )
    if original_overlay_path is not None:
        response.original_overlay_url = "/api/artifacts?%s" % urlencode(
            {"path": original_overlay_path},
        )
    return response
