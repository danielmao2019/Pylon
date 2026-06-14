"""Camera display response API."""

from base64 import b64encode
from json import dumps
from typing import Any, Dict, List, Optional

import torch

from data.structures.three_d.camera.camera_vis import cameras_vis
from data.structures.three_d.camera.cameras import Cameras
from data.viewer.utils.atomic_displays.cameras.ts.backend.schemas.display_response import (
    CameraDisplayResponse,
)


def create_camera_display_response(
    slot_id: str,
    title: str,
    cameras: Optional[Cameras],
) -> CameraDisplayResponse:
    """Create a camera display response.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        cameras: Caller-supplied Cameras collection to visualize, or None for no camera layer.

    Returns:
        Camera display response.
    """
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert cameras is None or isinstance(cameras, Cameras), (
        "Cameras must be None or a Cameras collection. cameras=%r" % cameras
    )

    camera_visualizations_url = None
    if cameras is not None:
        camera_visualizations = _build_camera_vis_payload(cameras=cameras)
        encoded_payload = b64encode(
            dumps(camera_visualizations, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        camera_visualizations_url = "data:application/json;base64,%s" % encoded_payload

    return CameraDisplayResponse(
        slot_id=slot_id,
        title=title,
        display_kind="camera",
        url=camera_visualizations_url,
        meta_info={},
    )


def _build_camera_vis_payload(cameras: Cameras) -> List[Dict[str, Any]]:
    """Build the JSON-compatible camera visualization payload.

    Args:
        cameras: Camera collection loaded from the selected camera artifact.

    Returns:
        JSON-compatible camera visualization payload.
    """
    assert isinstance(cameras, Cameras), (
        "Cameras must be a Cameras collection. cameras=%r" % cameras
    )
    return [
        _serialize_camera_vis_entry(camera_vis_entry=camera_vis_entry)
        for camera_vis_entry in cameras_vis(
            cameras=cameras,
            frustum_scale=0.25,
        )
    ]


def _serialize_camera_vis_entry(
    camera_vis_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Serialize one camera visualization entry.

    Args:
        camera_vis_entry: Camera visualization primitive dictionary.

    Returns:
        JSON-compatible camera visualization entry.
    """
    assert isinstance(camera_vis_entry, dict), (
        "Camera visualization entry must be a dictionary. "
        "camera_vis_entry=%r" % camera_vis_entry
    )
    return {
        "center": camera_vis_entry["center"].detach().cpu().tolist(),
        "center_color": camera_vis_entry["center_color"].detach().cpu().tolist(),
        "axes": [
            _serialize_camera_vis_line(camera_vis_line=line)
            for line in camera_vis_entry["axes"]
        ],
        "frustum_lines": [
            _serialize_camera_vis_line(camera_vis_line=line)
            for line in camera_vis_entry["frustum_lines"]
        ],
    }


def _serialize_camera_vis_line(
    camera_vis_line: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Serialize one camera visualization line segment.

    Args:
        camera_vis_line: Camera visualization line with start, end, and color tensors.

    Returns:
        JSON-compatible camera visualization line.
    """
    assert isinstance(camera_vis_line, dict), (
        "Camera visualization line must be a dictionary. "
        "camera_vis_line=%r" % camera_vis_line
    )
    return {
        "start": camera_vis_line["start"].detach().cpu().tolist(),
        "end": camera_vis_line["end"].detach().cpu().tolist(),
        "color": camera_vis_line["color"].detach().cpu().tolist(),
    }
