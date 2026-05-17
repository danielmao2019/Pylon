"""Video display response API."""

from typing import Optional
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.videos.ts.backend.schemas.display_response import (
    VideoDisplayResponse,
)


def create_video_display_response(
    slot_id: str,
    title: str,
    video_path: Optional[str],
) -> VideoDisplayResponse:
    """Create a video display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        video_path: Video artifact path.

    Returns:
        Video display response.
    """
    if video_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": video_path})
    return VideoDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="video",
        url=url,
        meta_info={},
    )
