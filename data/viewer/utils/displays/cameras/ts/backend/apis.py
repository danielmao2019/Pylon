"""Camera display response APIs."""

from typing import Any, Dict, List, Optional, Tuple

from data.structures.three_d.camera.camera_vis import cameras_vis
from data.structures.three_d.camera.cameras import Cameras
from data.viewer.utils.displays.cameras.ts.backend.core_camera_display import (
    create_camera_display_response_core,
)
from data.viewer.utils.displays.cameras.ts.backend.schemas.display_response import (
    CameraDisplayResponse,
)


def create_camera_display_response(
    slot_id: str,
    title: str,
    cameras: Optional[Cameras],
    frustum_size: Optional[float] = None,
    frustum_color: Optional[Tuple[int, int, int]] = None,
    point_size: Optional[float] = None,
    point_color: Optional[Tuple[int, int, int]] = None,
) -> CameraDisplayResponse:
    """Create a camera display response from a caller-supplied Cameras.

    The caller may override the baked glyph styles; otherwise each None resolves to
    the cameras_vis module-global default. When cameras is None there is no
    camera-vis payload and the response url is None.

    Args:
        slot_id: Stable display slot identifier.
        title: Display panel title.
        cameras: Optional Cameras collection to visualize, or None for no camera
            layer.
        frustum_size: World-unit frustum/axis size, forwarded untouched; None is
            resolved to the cameras_vis default downstream.
        frustum_color: RGB frustum-line color as a 0-255 int tuple, forwarded
            untouched; None is resolved to the cameras_vis default downstream.
        point_size: World-unit camera-center marker size, forwarded untouched; None
            is resolved to the cameras_vis default downstream.
        point_color: RGB center-point color as a 0-255 int tuple, forwarded
            untouched; None is resolved to the cameras_vis default downstream.

    Returns:
        Camera display response.
    """
    assert isinstance(slot_id, str), "Slot id must be a string. slot_id=%r" % slot_id
    assert isinstance(title, str), "Title must be a string. title=%r" % title
    assert cameras is None or isinstance(cameras, Cameras), (
        "Cameras must be None or a Cameras collection. cameras=%r" % cameras
    )
    assert frustum_size is None or isinstance(frustum_size, float), (
        "Frustum size must be None or a float. frustum_size=%r" % frustum_size
    )
    assert frustum_color is None or isinstance(frustum_color, tuple), (
        "Frustum color must be None or a tuple. frustum_color=%r" % frustum_color
    )
    assert point_size is None or isinstance(point_size, float), (
        "Point size must be None or a float. point_size=%r" % point_size
    )
    assert point_color is None or isinstance(point_color, tuple), (
        "Point color must be None or a tuple. point_color=%r" % point_color
    )

    camera_vis_payload: Optional[List[Dict[str, Any]]] = None
    if cameras is not None:
        camera_vis_payload = _map_camera_params_to_vis(
            cameras=cameras,
            frustum_size=frustum_size,
            frustum_color=frustum_color,
            point_size=point_size,
            point_color=point_color,
        )

    return create_camera_display_response_core(
        slot_id=slot_id,
        title=title,
        camera_vis_payload=camera_vis_payload,
        meta_info={},
    )


def _map_camera_params_to_vis(
    cameras: Cameras,
    frustum_size: Optional[float],
    frustum_color: Optional[Tuple[int, int, int]],
    point_size: Optional[float],
    point_color: Optional[Tuple[int, int, int]],
) -> List[Dict[str, Any]]:
    """Map a Cameras collection to the JSON-able camera-vis payload.

    Applies the caller's baked styles or their cameras_vis defaults; the camera
    sibling of _map_segmentation_pc_to_rgb.

    Args:
        cameras: Camera collection loaded from the selected camera artifact.
        frustum_size: World-unit frustum/axis size, forwarded untouched.
        frustum_color: RGB frustum-line color as a 0-255 int tuple, forwarded
            untouched.
        point_size: World-unit camera-center marker size, forwarded untouched.
        point_color: RGB center-point color as a 0-255 int tuple, forwarded
            untouched.

    Returns:
        JSON-able list of serialized camera-vis entries.
    """
    assert isinstance(cameras, Cameras), (
        "Cameras must be a Cameras collection. cameras=%r" % cameras
    )
    assert frustum_size is None or isinstance(frustum_size, float), (
        "Frustum size must be None or a float. frustum_size=%r" % frustum_size
    )
    assert frustum_color is None or isinstance(frustum_color, tuple), (
        "Frustum color must be None or a tuple. frustum_color=%r" % frustum_color
    )
    assert point_size is None or isinstance(point_size, float), (
        "Point size must be None or a float. point_size=%r" % point_size
    )
    assert point_color is None or isinstance(point_color, tuple), (
        "Point color must be None or a tuple. point_color=%r" % point_color
    )

    return [
        _serialize_camera_vis_entry(camera_vis_entry=camera_vis_entry)
        for camera_vis_entry in cameras_vis(
            cameras=cameras,
            frustum_size=frustum_size,
            frustum_color=frustum_color,
            point_size=point_size,
            point_color=point_color,
        )
    ]


def _serialize_camera_vis_entry(camera_vis_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one camera-vis entry into the JSON shape consumed by the renderer.

    Args:
        camera_vis_entry: Camera-vis primitive dict with keys center, center_color,
            center_size, axes, and frustum_lines.

    Returns:
        JSON-able camera-vis entry.
    """
    assert isinstance(camera_vis_entry, dict), (
        "Camera vis entry must be a dictionary. camera_vis_entry=%r" % camera_vis_entry
    )
    return {
        "center": camera_vis_entry["center"].detach().cpu().tolist(),
        "center_color": camera_vis_entry["center_color"].detach().cpu().tolist(),
        "center_size": camera_vis_entry["center_size"],
        "axes": [
            _serialize_camera_vis_line(camera_vis_line=line)
            for line in camera_vis_entry["axes"]
        ],
        "frustum_lines": [
            _serialize_camera_vis_line(camera_vis_line=line)
            for line in camera_vis_entry["frustum_lines"]
        ],
    }


def _serialize_camera_vis_line(camera_vis_line: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one camera-vis line segment into plain start, end, and color lists.

    Args:
        camera_vis_line: Camera-vis line with start, end, and color tensors.

    Returns:
        JSON-able camera-vis line.
    """
    assert isinstance(camera_vis_line, dict), (
        "Camera vis line must be a dictionary. camera_vis_line=%r" % camera_vis_line
    )
    return {
        "start": camera_vis_line["start"].detach().cpu().tolist(),
        "end": camera_vis_line["end"].detach().cpu().tolist(),
        "color": camera_vis_line["color"].detach().cpu().tolist(),
    }
