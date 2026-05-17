"""Pixel display response core."""

from typing import Any, Dict, Optional, Type
from urllib.parse import urlencode

from data.viewer.utils.atomic_displays.pixels.ts.backend.schemas.display_response import (
    PixelDisplayResponse,
)


def create_pixels_display_response(
    response_type: Type[PixelDisplayResponse],
    slot_id: str,
    title: str,
    display_kind: str,
    image_path: Optional[str],
    meta_info: Dict[str, Any] | None = None,
) -> PixelDisplayResponse:
    """Create a pixel display response.

    Args:
        response_type: Concrete pixel display response model.
        slot_id: Stable display slot identifier.
        title: Display panel title.
        display_kind: Atomic display kind.
        image_path: Image artifact path.
        meta_info: Optional renderer metadata.

    Returns:
        Pixel display response.
    """
    assert issubclass(
        response_type, PixelDisplayResponse
    ), "Response type must be a PixelDisplayResponse subclass. response_type=%r" % (
        response_type,
    )
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert isinstance(display_kind, str), (
        "Display kind must be a string. display_kind=%r" % display_kind
    )
    assert image_path is None or isinstance(image_path, str), (
        "Image path must be None or a string. image_path=%r" % image_path
    )
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )

    if image_path is None:
        url: Optional[str] = None
    else:
        url = "/api/artifacts?%s" % urlencode({"path": image_path})

    return response_type(
        slot_id=slot_id,
        title=title,
        display_kind=display_kind,
        url=url,
        meta_info={} if meta_info is None else dict(meta_info),
    )
