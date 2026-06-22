"""2D axis-aligned-box display response schema."""

from typing import List, Literal, Optional

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class Aabb2dDisplayResponse(DisplayResponse):
    """2D axis-aligned-box overlay display response.

    Raster overlay response carrying inline axis-aligned 2D boxes (each a 4-float
    box ``[min_x, min_y, max_x, max_y]``) with optional per-box scores, composed
    as an aux layer over an image.

    Args:
        None.

    Returns:
        Pydantic model for 2D axis-aligned-box overlay displays.
    """

    display_kind: Literal["aabb_2d"] = "aabb_2d"
    aabbs: List[List[float]]
    scores: Optional[List[float]] = None
