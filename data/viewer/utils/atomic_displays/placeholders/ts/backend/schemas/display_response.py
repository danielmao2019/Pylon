"""Placeholder display response schema."""

from typing import Literal

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class PlaceholderDisplayResponse(DisplayResponse):
    """Missing-result placeholder display response.

    Args:
        None.

    Returns:
        Pydantic model for placeholder display responses.
    """

    display_kind: Literal["placeholder"] = "placeholder"
    message: str
