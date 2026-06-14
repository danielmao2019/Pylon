"""Camera display response schema."""

from typing import Literal

from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class CameraDisplayResponse(DisplayResponse):
    """Camera display response.

    Args:
        None.

    Returns:
        Pydantic model for camera displays.
    """

    display_kind: Literal["camera"] = "camera"
