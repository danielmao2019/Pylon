"""Point display response core."""

from typing import Any, Dict, Optional, Type
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.points.ts.backend.schemas.display_response import (
    PointDisplayResponse,
)


def create_points_display_response_core(
    response_type: Type[PointDisplayResponse],
    slot_id: str,
    title: str,
    display_kind: str,
    point_cloud_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> PointDisplayResponse:
    """Create a point display response.

    Args:
        response_type: Concrete point display response model.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        display_kind: Atomic display kind.
        point_cloud_path: Point-cloud artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Point display response.
    """
    assert issubclass(
        response_type, PointDisplayResponse
    ), "Response type must be a PointDisplayResponse subclass. response_type=%r" % (
        response_type,
    )
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert isinstance(display_kind, str), (
        "Display kind must be a string. display_kind=%r" % display_kind
    )
    assert point_cloud_path is None or isinstance(point_cloud_path, str), (
        "Point cloud path must be None or a string. point_cloud_path=%r"
        % point_cloud_path
    )
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )
    assert title, "Title must be non-empty. title=%r" % title

    if point_cloud_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": point_cloud_path})

    return response_type(
        slot_id=slot_id,
        title=title,
        display_kind=display_kind,
        url=url,
        meta_info={} if meta_info is None else dict(meta_info),
    )
