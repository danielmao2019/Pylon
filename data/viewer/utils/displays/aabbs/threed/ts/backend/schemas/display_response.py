"""3D axis-aligned-box display response schema."""

from typing import List, Literal, Optional

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class Aabb3dDisplayResponse(DisplayResponse):
    """3D axis-aligned-box overlay display response.

    Spatial overlay response carrying inline axis-aligned 3D boxes (each a 6-float
    box ``[min_x, min_y, min_z, max_x, max_y, max_z]``) with optional per-box
    scores, composed as an aux layer over a point cloud.

    Args:
        None.

    Returns:
        Pydantic model for 3D axis-aligned-box overlay displays.
    """

    display_kind: Literal["aabb_3d"] = "aabb_3d"
    aabbs: List[List[float]]
    scores: Optional[List[float]] = None
