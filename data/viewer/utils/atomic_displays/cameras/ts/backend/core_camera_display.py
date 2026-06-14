"""Camera display response core."""

from base64 import b64encode
from json import dumps
from typing import Any, Dict, List, Optional

from data.viewer.utils.atomic_displays.cameras.ts.backend.schemas.display_response import (
    CameraDisplayResponse,
)


def create_camera_display_response_core(
    slot_id: str,
    title: str,
    camera_vis_payload: Optional[List[Dict[str, Any]]],
    meta_info: Optional[Dict[str, Any]] = None,
) -> CameraDisplayResponse:
    """Create a camera display response from the already-mapped camera-vis payload.

    Exposes the payload through a frontend-loadable base64 data URL. When
    ``camera_vis_payload`` is ``None`` or empty, no camera-vis resource exists and
    the response ``url`` is ``None``.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        camera_vis_payload: JSON-able list of serialized camera-vis entries, or
            None when there is no camera artifact to visualize.
        meta_info: Optional renderer metadata; an empty object for camera display.

    Returns:
        Camera display response.
    """
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert camera_vis_payload is None or isinstance(camera_vis_payload, list), (
        "Camera vis payload must be None or a list. camera_vis_payload=%r"
        % camera_vis_payload
    )
    assert meta_info is None or isinstance(meta_info, dict), (
        "Meta info must be None or a dict. meta_info=%r" % meta_info
    )

    if not camera_vis_payload:
        url: Optional[str] = None
    else:
        encoded_payload = b64encode(
            dumps(camera_vis_payload, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        url = "data:application/json;base64,%s" % encoded_payload

    return CameraDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="camera",
        url=url,
        meta_info={} if meta_info is None else dict(meta_info),
    )
