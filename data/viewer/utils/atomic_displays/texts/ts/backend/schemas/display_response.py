"""Text display response schema."""

from typing import Literal

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class TextDisplayResponse(DisplayResponse):
    """Text display response.

    Args:
        None.

    Returns:
        Pydantic model for text displays.
    """

    display_kind: Literal["text"] = "text"
    text: str
