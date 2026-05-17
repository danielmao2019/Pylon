"""Mesh display response APIs."""

from typing import Any, Dict, Optional
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.mesh.ts.backend.core_mesh_display import (
    create_mesh_display_response,
)
from data.viewer.utils.atomic_displays.mesh.ts.backend.schemas.display_response import (
    ColorMeshDisplayResponse,
    SegmentationMeshDisplayResponse,
)


def create_color_mesh_display_response(
    slot_id: str,
    title: str,
    mesh_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> ColorMeshDisplayResponse:
    """Create a color mesh display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        mesh_path: Mesh artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Color mesh display response.
    """
    return create_mesh_display_response(
        response_type=ColorMeshDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="color_mesh",
        mesh_path=mesh_path,
        meta_info=meta_info,
    )


def create_segmentation_mesh_display_response(
    slot_id: str,
    title: str,
    segmentation_mesh_path: Optional[str],
    original_overlay_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> SegmentationMeshDisplayResponse:
    """Create a segmentation mesh display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        segmentation_mesh_path: Segmentation mesh artifact path.
        original_overlay_path: Original scene artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Segmentation mesh display response.
    """
    payload = {} if meta_info is None else dict(meta_info)
    response = create_mesh_display_response(
        response_type=SegmentationMeshDisplayResponse,
        slot_id=slot_id,
        title=title,
        display_kind="segmentation_mesh",
        mesh_path=segmentation_mesh_path,
        meta_info=payload,
    )
    if original_overlay_path is not None:
        response.original_overlay_url = "/api/artifacts?%s" % urlencode(
            {"path": original_overlay_path},
        )
    return response
