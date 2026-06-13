"""Placeholder display response API."""

from data.viewer.utils.displays.placeholders.ts.backend.schemas.display_response import (
    PlaceholderDisplayResponse,
)


def create_placeholder_display_response(
    slot_id: str,
    title: str,
    message: str,
) -> PlaceholderDisplayResponse:
    """Create a missing-result placeholder display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        message: Placeholder message.

    Returns:
        Placeholder display response.
    """
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert isinstance(message, str), "Message must be a string. message=%r" % message
    return PlaceholderDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="placeholder",
        url=None,
        meta_info={},
        message=message,
    )
