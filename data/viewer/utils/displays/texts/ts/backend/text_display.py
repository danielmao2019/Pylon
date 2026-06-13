"""Text display response API."""

from data.viewer.utils.displays.texts.ts.backend.schemas.display_response import (
    TextDisplayResponse,
)


def create_text_display_response(
    slot_id: str,
    title: str,
    text: str,
) -> TextDisplayResponse:
    """Create a text display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        text: Text content to render.

    Returns:
        Text display response.
    """
    assert isinstance(text, str), "Text must be a string. text=%r" % text
    return TextDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="text",
        url=None,
        meta_info={},
        text=text,
    )
