"""Layered display response schema."""

from typing import List, Literal

from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import (
    DisplayResponse,
)


class LayeredDisplayResponse(DisplayResponse):
    """Layered display response.

    Args:
        None.

    Returns:
        Pydantic model carrying one base display response plus an ordered list
        of auxiliary display responses stacked on top of the base. Consumers
        own per-layer semantics and visibility state.
    """

    display_kind: Literal["layered"] = "layered"
    base_display_response: DisplayResponse
    aux_display_responses: List[DisplayResponse]
