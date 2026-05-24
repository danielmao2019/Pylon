"""Point display response schemas."""

from typing import Literal, Optional

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class PointDisplayResponse(DisplayResponse):
    """Base point display response.

    Args:
        None.

    Returns:
        Pydantic model for point display responses.
    """


class ColorPCDisplayResponse(PointDisplayResponse):
    """Color point-cloud display response.

    Args:
        None.

    Returns:
        Pydantic model for color point-cloud displays.
    """

    display_kind: Literal["color_pc"] = "color_pc"


class SegmentationPCDisplayResponse(PointDisplayResponse):
    """Segmentation point-cloud display response.

    Args:
        None.

    Returns:
        Pydantic model for segmentation point-cloud displays.
    """

    display_kind: Literal["segmentation_pc"] = "segmentation_pc"
    original_overlay_url: Optional[str] = None
