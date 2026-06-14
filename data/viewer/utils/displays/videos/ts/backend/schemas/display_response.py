"""Video display response schema."""

from typing import Literal

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class VideoDisplayResponse(DisplayResponse):
    """Video display response.

    Args:
        None.

    Returns:
        Pydantic model for video displays.
    """

    display_kind: Literal["video"] = "video"
