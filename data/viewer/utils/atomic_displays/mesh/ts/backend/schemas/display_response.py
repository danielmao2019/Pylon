"""Mesh display response schemas."""

from typing import Literal, Optional

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class MeshDisplayResponse(DisplayResponse):
    """Base mesh display response.

    Args:
        None.

    Returns:
        Pydantic model for mesh display responses.
    """


class ColorMeshDisplayResponse(MeshDisplayResponse):
    """Color mesh display response.

    Args:
        None.

    Returns:
        Pydantic model for color mesh displays.
    """

    display_kind: Literal["color_mesh"] = "color_mesh"


class SegmentationMeshDisplayResponse(MeshDisplayResponse):
    """Segmentation mesh display response.

    Args:
        None.

    Returns:
        Pydantic model for segmentation mesh displays.
    """

    display_kind: Literal["segmentation_mesh"] = "segmentation_mesh"
    original_overlay_url: Optional[str] = None
