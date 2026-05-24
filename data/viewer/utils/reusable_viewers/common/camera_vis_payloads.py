"""Shared camera-visualization payload helpers for viewer tasks.

Args:
    None.

Returns:
    None.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.camera_vis import camera_vis
from data.structures.three_d.camera.cameras import Cameras


def create_camera_visualization_payload(
    camera_poses_path: Path,
    intrinsics_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Create camera-visualization payloads from disk camera resources.

    Args:
        camera_poses_path: Path to OpenCV `c2w` pose rows, one 4x4 matrix per row.
        intrinsics_path: Optional path to `fx fy cx cy` intrinsic rows.

    Returns:
        JSON-serializable camera-visualization entries.
    """

    assert isinstance(camera_poses_path, Path), (
        "Camera poses path must be a Path. "
        f"camera_poses_path_type={type(camera_poses_path)}."
    )
    assert intrinsics_path is None or isinstance(intrinsics_path, Path), (
        "Intrinsics path must be a Path or `None`. "
        f"intrinsics_path_type={type(intrinsics_path)}."
    )

    cameras = load_opencv_c2w_cameras(camera_poses_path=camera_poses_path)
    viewer_intrinsics = (
        None
        if intrinsics_path is None
        else load_average_intrinsics(intrinsics_path=intrinsics_path)
    )
    return serialize_camera_vis_list(
        cameras=cameras,
        viewer_intrinsics=viewer_intrinsics,
        axis_length=0.15,
        frustum_depth=0.25,
    )


def load_opencv_c2w_cameras(camera_poses_path: Path) -> Cameras:
    """Load OpenCV camera-to-world pose rows as shared camera objects.

    Args:
        camera_poses_path: Path to OpenCV `c2w` pose rows, one 4x4 matrix per row.

    Returns:
        Shared `Cameras` collection on CPU.
    """

    assert isinstance(camera_poses_path, Path), (
        "Camera poses path must be a Path. "
        f"camera_poses_path_type={type(camera_poses_path)}."
    )
    assert camera_poses_path.is_file(), (
        "Camera poses path must be an existing file. "
        f"camera_poses_path={camera_poses_path}."
    )

    extrinsics_list: List[torch.Tensor] = []
    with camera_poses_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if stripped_line == "":
                continue
            row_values = stripped_line.split()
            assert len(row_values) == 16, (
                "Camera pose rows must contain 16 serialized floats. "
                f"line_index={line_index} num_values={len(row_values)} "
                f"camera_poses_path={camera_poses_path}."
            )
            camera_to_world = torch.tensor(
                [float(value) for value in row_values],
                dtype=torch.float32,
            ).reshape(4, 4)
            camera_to_world[:3, :3] = _stabilize_rotation(
                rotation=camera_to_world[:3, :3],
            )
            extrinsics_list.append(camera_to_world)

    assert len(extrinsics_list) > 0, (
        "Camera poses path must contain at least one pose row. "
        f"camera_poses_path={camera_poses_path}."
    )
    return Cameras(
        intrinsics=None,
        extrinsics=extrinsics_list,
        conventions=["opencv"] * len(extrinsics_list),
        device="cpu",
    )


def load_average_intrinsics(intrinsics_path: Path) -> torch.Tensor:
    """Load average pixel intrinsics for camera frustum shaping.

    Args:
        intrinsics_path: Path to `fx fy cx cy` intrinsic rows.

    Returns:
        `float32 [3, 3]` average intrinsics matrix.
    """

    assert isinstance(intrinsics_path, Path), (
        "Intrinsics path must be a Path. "
        f"intrinsics_path_type={type(intrinsics_path)}."
    )
    assert intrinsics_path.is_file(), (
        "Intrinsics path must be an existing file. "
        f"intrinsics_path={intrinsics_path}."
    )

    intrinsic_rows: List[List[float]] = []
    with intrinsics_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            stripped_line = line.strip()
            if stripped_line == "":
                continue
            row_values = stripped_line.split()
            assert len(row_values) == 4, (
                "Intrinsic rows must contain `fx fy cx cy`. "
                f"line_index={line_index} num_values={len(row_values)} "
                f"intrinsics_path={intrinsics_path}."
            )
            intrinsic_rows.append([float(value) for value in row_values])

    assert len(intrinsic_rows) > 0, (
        "Intrinsics path must contain at least one row. "
        f"intrinsics_path={intrinsics_path}."
    )
    average_row = torch.tensor(intrinsic_rows, dtype=torch.float32).mean(dim=0)
    return torch.tensor(
        [
            [float(average_row[0]), 0.0, float(average_row[2])],
            [0.0, float(average_row[1]), float(average_row[3])],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def serialize_camera_vis_entry(
    camera: Camera,
    viewer_intrinsics: Optional[torch.Tensor] = None,
    axis_length: float = 0.15,
    frustum_depth: float = 0.25,
) -> Dict[str, Any]:
    """Serialize one shared `camera_vis()` payload to JSON-compatible data.

    Args:
        camera: Shared `Camera` instance whose basis defines the visualization.
        viewer_intrinsics: Optional `float32 [3, 3]` pixel intrinsics for frustum shape.
        axis_length: Axis length in world units.
        frustum_depth: Frustum depth in world units.

    Returns:
        JSON-serializable camera-visualization dict.
    """

    assert isinstance(camera, Camera), (
        "Camera visualization input must be a shared `Camera`. "
        f"camera_type={type(camera)}."
    )
    assert viewer_intrinsics is None or isinstance(viewer_intrinsics, torch.Tensor), (
        "Viewer intrinsics must be a tensor or `None`. "
        f"viewer_intrinsics_type={type(viewer_intrinsics)}."
    )
    assert isinstance(axis_length, float), (
        "Axis length must be a float. " f"axis_length_type={type(axis_length)}."
    )
    assert isinstance(frustum_depth, float), (
        "Frustum depth must be a float. " f"frustum_depth_type={type(frustum_depth)}."
    )

    if viewer_intrinsics is None:
        vis_entry = camera_vis(
            camera=camera,
            axis_length=axis_length,
            frustum_depth=frustum_depth,
            frustum_color=None,
        )
    else:
        vis_entry = _build_camera_vis_entry_with_intrinsics(
            camera=camera,
            viewer_intrinsics=viewer_intrinsics,
            axis_length=axis_length,
            frustum_depth=frustum_depth,
        )
    return _to_jsonable_camera_vis_entry(vis_entry=vis_entry)


def serialize_camera_vis_list(
    cameras: Cameras,
    viewer_intrinsics: Optional[torch.Tensor] = None,
    axis_length: float = 0.15,
    frustum_depth: float = 0.25,
) -> List[Dict[str, Any]]:
    """Serialize a camera trajectory to JSON-compatible camera-visualization data.

    Args:
        cameras: Shared `Cameras` collection in OpenCV `c2w` convention.
        viewer_intrinsics: Optional `float32 [3, 3]` pixel intrinsics for frustum shape.
        axis_length: Axis length in world units.
        frustum_depth: Frustum depth in world units.

    Returns:
        List of JSON-serializable camera-visualization dicts.
    """

    assert isinstance(cameras, Cameras), (
        "Camera trajectory must be a shared `Cameras` collection. "
        f"cameras_type={type(cameras)}."
    )

    vis_entries: List[Dict[str, Any]] = []
    for camera in cameras:
        vis_entries.append(
            serialize_camera_vis_entry(
                camera=camera,
                viewer_intrinsics=viewer_intrinsics,
                axis_length=axis_length,
                frustum_depth=frustum_depth,
            )
        )
    return vis_entries


def serialize_first_camera_pose(
    cameras: Cameras,
) -> Dict[str, Any]:
    """Serialize the first camera pose from one trajectory.

    Args:
        cameras: Shared `Cameras` collection in OpenCV `c2w` convention.

    Returns:
        JSON-serializable first-camera payload.
    """

    assert isinstance(cameras, Cameras), (
        "Camera trajectory must be a shared `Cameras` collection. "
        f"cameras_type={type(cameras)}."
    )
    assert len(cameras) > 0, (
        "First-camera payload requires a non-empty trajectory. "
        f"num_cameras={len(cameras)}."
    )
    first_camera = cameras[0]
    assert isinstance(first_camera, Camera), (
        "First camera entry must be a shared `Camera`. "
        f"first_camera_type={type(first_camera)}."
    )
    return {
        "extrinsics": first_camera.extrinsics.tolist(),
        "extrinsics_format": "c2w",
        "convention": "opencv",
    }


def _to_jsonable_camera_vis_entry(
    vis_entry: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert one shared camera-vis payload into plain Python containers.

    Args:
        vis_entry: Camera-vis payload returned by the shared primitive.

    Returns:
        JSON-serializable camera-vis dict.
    """

    assert isinstance(vis_entry, dict), (
        "Camera-vis payload must be a dict. " f"vis_entry_type={type(vis_entry)}."
    )
    assert "center" in vis_entry, (
        "Camera-vis payload must include a center. "
        f"vis_entry_keys={sorted(vis_entry.keys())}."
    )
    assert "center_color" in vis_entry, (
        "Camera-vis payload must include a center color. "
        f"vis_entry_keys={sorted(vis_entry.keys())}."
    )
    assert "axes" in vis_entry, (
        "Camera-vis payload must include axes. "
        f"vis_entry_keys={sorted(vis_entry.keys())}."
    )
    assert "frustum_lines" in vis_entry, (
        "Camera-vis payload must include frustum lines. "
        f"vis_entry_keys={sorted(vis_entry.keys())}."
    )

    return {
        "center": vis_entry["center"].detach().cpu().tolist(),
        "center_color": vis_entry["center_color"].detach().cpu().tolist(),
        "axes": [_line_to_dict(line=line) for line in vis_entry["axes"]],
        "frustum_lines": [
            _line_to_dict(line=line) for line in vis_entry["frustum_lines"]
        ],
    }


def _build_camera_vis_entry_with_intrinsics(
    camera: Camera,
    viewer_intrinsics: torch.Tensor,
    axis_length: float,
    frustum_depth: float,
) -> Dict[str, Any]:
    """Build a camera-vis payload with viewer-specific intrinsics shaping.

    Args:
        camera: Shared `Camera` instance whose basis defines the visualization.
        viewer_intrinsics: `float32 [3, 3]` pixel intrinsics for frustum shape.
        axis_length: Axis length in world units.
        frustum_depth: Frustum depth in world units.

    Returns:
        JSON-serializable camera-vis dict.
    """

    assert isinstance(viewer_intrinsics, torch.Tensor), (
        "Viewer intrinsics must be a tensor when shaping the frustum. "
        f"viewer_intrinsics_type={type(viewer_intrinsics)}."
    )
    assert viewer_intrinsics.shape == (3, 3), (
        "Viewer intrinsics must have shape `(3, 3)`. "
        f"viewer_intrinsics_shape={tuple(viewer_intrinsics.shape)}."
    )

    assert isinstance(axis_length, float), (
        "Axis length must be a float. " f"axis_length_type={type(axis_length)}."
    )
    assert isinstance(frustum_depth, float), (
        "Frustum depth must be a float. " f"frustum_depth_type={type(frustum_depth)}."
    )
    assert axis_length > 0.0, (
        "Axis length must be positive. " f"axis_length={axis_length}."
    )
    assert frustum_depth > 0.0, (
        "Frustum depth must be positive. " f"frustum_depth={frustum_depth}."
    )
    assert viewer_intrinsics.dtype == torch.float32, (
        "Viewer intrinsics must be float32 before camera-vis construction. "
        f"viewer_intrinsics_dtype={viewer_intrinsics.dtype}."
    )

    device = camera.device
    dtype = camera.center.dtype
    camera_center = camera.center
    camera_right = camera.right
    camera_forward = camera.forward
    camera_up = camera.up
    axes = [
        {
            "start": camera_center,
            "end": camera_center + camera_right * axis_length,
            "color": torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype),
        },
        {
            "start": camera_center,
            "end": camera_center + camera_forward * axis_length,
            "color": torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype),
        },
        {
            "start": camera_center,
            "end": camera_center + camera_up * axis_length,
            "color": torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype),
        },
    ]
    frustum_color = torch.tensor([1.0, 0.84, 0.0], device=device, dtype=dtype)
    normalized_intrinsics = viewer_intrinsics.to(device=device, dtype=dtype)
    fx = normalized_intrinsics[0, 0]
    fy = normalized_intrinsics[1, 1]
    cx = normalized_intrinsics[0, 2]
    cy = normalized_intrinsics[1, 2]
    assert float(fx) > 0.0 and float(fy) > 0.0, (
        "Viewer focal lengths must stay positive. " f"fx={float(fx)} fy={float(fy)}."
    )
    assert float(cx) >= 0.0 and float(cy) >= 0.0, (
        "Viewer principal point must stay non-negative. "
        f"cx={float(cx)} cy={float(cy)}."
    )
    frustum_half_width = frustum_depth * (cx / fx)
    frustum_half_height = frustum_depth * (cy / fy)
    frustum_center = camera_center + camera_forward * frustum_depth
    frustum_points_world = [
        frustum_center
        - camera_right * frustum_half_width
        + camera_up * frustum_half_height,
        frustum_center
        + camera_right * frustum_half_width
        + camera_up * frustum_half_height,
        frustum_center
        + camera_right * frustum_half_width
        - camera_up * frustum_half_height,
        frustum_center
        - camera_right * frustum_half_width
        - camera_up * frustum_half_height,
    ]
    frustum_lines: List[Dict[str, torch.Tensor]] = []
    for point in frustum_points_world:
        frustum_lines.append(
            {"start": camera.center, "end": point, "color": frustum_color}
        )
    for index in range(len(frustum_points_world)):
        frustum_lines.append(
            {
                "start": frustum_points_world[index],
                "end": frustum_points_world[(index + 1) % len(frustum_points_world)],
                "color": frustum_color,
            }
        )
    return {
        "center": camera_center,
        "axes": axes,
        "frustum_lines": frustum_lines,
        "center_color": torch.tensor([0.0, 0.0, 0.0], device=device, dtype=dtype),
    }


def _line_to_dict(
    line: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Convert one shared camera-vis line segment to JSON.

    Args:
        line: Camera-vis line dict with `start`, `end`, and `color` tensors.

    Returns:
        JSON-serializable line dict.
    """

    assert isinstance(line, dict), (
        "Camera-vis line must be a dict. " f"line_type={type(line)}."
    )
    for key in ["start", "end", "color"]:
        assert key in line, (
            "Camera-vis line is missing a required key. "
            f"key={key!r} line_keys={sorted(line.keys())}."
        )
        assert isinstance(line[key], torch.Tensor), (
            "Camera-vis line values must be torch tensors. "
            f"key={key!r} value_type={type(line[key])}."
        )
    return {
        "start": line["start"].detach().cpu().tolist(),
        "end": line["end"].detach().cpu().tolist(),
        "color": line["color"].detach().cpu().tolist(),
    }


def _stabilize_rotation(rotation: torch.Tensor) -> torch.Tensor:
    """Project one near-rotation matrix onto `SO(3)`.

    Args:
        rotation: `float32 [3, 3]` matrix expected to be close to orthogonal.

    Returns:
        Proper `float32 [3, 3]` rotation matrix with determinant `+1`.
    """

    assert isinstance(rotation, torch.Tensor), (
        "Rotation matrix must be a torch tensor. " f"rotation_type={type(rotation)}."
    )
    assert rotation.shape == (3, 3), (
        "Rotation matrix must have shape `(3, 3)`. "
        f"rotation_shape={tuple(rotation.shape)}."
    )
    assert rotation.dtype == torch.float32, (
        "Rotation matrix must be float32. " f"rotation_dtype={rotation.dtype}."
    )

    rotation_float64 = rotation.to(dtype=torch.float64)
    u_matrix, _, v_transpose = torch.linalg.svd(rotation_float64)
    stabilized_rotation = u_matrix @ v_transpose
    if float(torch.linalg.det(stabilized_rotation)) < 0.0:
        u_matrix[:, -1] = -1.0 * u_matrix[:, -1]
        stabilized_rotation = u_matrix @ v_transpose
    return stabilized_rotation.to(dtype=torch.float32)
