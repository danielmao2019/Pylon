"""Shared camera-space geometry helpers for mesh texture extraction."""

from typing import Tuple

import nvdiffrast.torch as dr
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.camera.intrinsics.camera_intrinsics import CameraIntrinsics
from models.three_d.point_cloud.ops.world_to_camera_transform import (
    world_to_camera_transform,
)

PERSPECTIVE_DEPTH_FLOOR = 1e-8


def render_camera_face_index_buffer(
    verts_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: CameraIntrinsics,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Render a one-view camera-space face-index buffer.

    Args:
        verts_camera: Camera-space verts [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: The view's CameraIntrinsics, whose own model owns the projection.
        image_height: Render height in pixels.
        image_width: Render width in pixels.

    Returns:
        Face-index image [1, H, W, 1] with values face_index + 1 and 0 for background.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(verts_camera, torch.Tensor), f"{type(verts_camera)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(intrinsics, CameraIntrinsics), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert verts_camera.ndim == 2, f"{verts_camera.shape=}"
        assert verts_camera.shape[1] == 3, f"{verts_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    clip_verts = _camera_verts_to_clip(
        verts_camera=verts_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=verts_camera.device, dtype=torch.float32)
    tri_i32 = faces.to(device=verts_camera.device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=verts_camera.device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_verts.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )

    face_indices = rast_out[..., 3].to(dtype=torch.long) - 1
    face_plus1 = (face_indices + 1).to(dtype=torch.float32).unsqueeze(-1)
    visible = face_indices >= 0
    return torch.where(visible.unsqueeze(-1), face_plus1, torch.zeros_like(face_plus1))


def render_camera_depth_buffer(
    verts_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: CameraIntrinsics,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Render a one-view camera-space depth buffer.

    Args:
        verts_camera: Camera-space verts [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: The view's CameraIntrinsics, whose own model owns the projection.
        image_height: Render height in pixels.
        image_width: Render width in pixels.

    Returns:
        Depth image [1, H, W, 1] in camera-space z units with zeros for background.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(verts_camera, torch.Tensor), f"{type(verts_camera)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(intrinsics, CameraIntrinsics), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert verts_camera.ndim == 2, f"{verts_camera.shape=}"
        assert verts_camera.shape[1] == 3, f"{verts_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    clip_verts = _camera_verts_to_clip(
        verts_camera=verts_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=verts_camera.device, dtype=torch.float32)
    tri_i32 = faces.to(device=verts_camera.device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=verts_camera.device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_verts.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )
    depth_image, _ = dr.interpolate(
        attr=verts_camera[:, 2:3].unsqueeze(0).contiguous(),
        rast=rast_out,
        tri=tri_i32,
    )
    visible = rast_out[..., 3] > 0
    return torch.where(
        visible.unsqueeze(-1),
        depth_image,
        torch.zeros_like(depth_image),
    )


def _verts_world_to_camera(
    verts: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Transform one-view world-space verts to camera-space verts.

    Args:
        verts: Mesh verts in world coordinates [V, 3].
        camera: One camera instance.

    Returns:
        Camera-space verts [V, 3].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(verts, torch.Tensor), f"{type(verts)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert verts.ndim == 2, f"{verts.shape=}"
        assert verts.shape[1] == 3, f"{verts.shape=}"
        assert len(camera) == 1, f"{len(camera)=}"

    _validate_inputs()

    camera_single = camera[0].to(device=verts.device, extr_convention="opencv")
    verts_camera = world_to_camera_transform(
        points=verts,
        extrinsics=camera_single.extrinsics.extrinsics,
        inplace=False,
    )
    assert isinstance(verts_camera, torch.Tensor), f"{type(verts_camera)=}"
    assert verts_camera.shape == verts.shape, f"{verts_camera.shape=} {verts.shape=}"
    return verts_camera


def project_verts_to_image(
    verts: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space verts to image pixels for one view.

    Args:
        verts: Mesh verts in world coordinates [V, 3].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        A tuple of:
            xy: Pixel coordinates [V, 2].
            depth: Camera-space depth [V].
            verts_camera: Camera-space verts [V, 3].
            valid: In-frame projection validity mask [V].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(verts, torch.Tensor), f"{type(verts)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert verts.ndim == 2, f"{verts.shape=}"
        assert verts.shape[1] == 3, f"{verts.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert len(camera) == 1, f"{len(camera)=}"

    _validate_inputs()

    camera_single = camera[0].to(device=verts.device, extr_convention="opencv")
    verts_camera = _verts_world_to_camera(
        verts=verts,
        camera=camera,
    )
    depth = verts_camera[:, 2]

    xy = camera_single.intrinsics.project(points_camera=verts_camera, inplace=False)
    in_front = compute_points_in_front_of_camera(
        points_camera=verts_camera,
        intrinsics=camera_single.intrinsics,
    )
    valid = (
        in_front
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= float(image_width - 1))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= float(image_height - 1))
    )
    return xy, depth, verts_camera, valid


def compute_points_in_front_of_camera(
    points_camera: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    """Mark which camera-space points the camera's own model can see.

    Args:
        points_camera: Camera-space points [N, 3].
        intrinsics: The view's CameraIntrinsics, whose model decides what "in front" means.

    Returns:
        Bool mask [N] over `points_camera`'s rows.
    """

    def _validate_inputs() -> None:
        # Input validations
        assert isinstance(points_camera, torch.Tensor), f"{type(points_camera)=}"
        assert isinstance(intrinsics, CameraIntrinsics), f"{type(intrinsics)=}"
        assert points_camera.ndim == 2, f"{points_camera.shape=}"
        assert points_camera.shape[1] == 3, f"{points_camera.shape=}"

    _validate_inputs()

    if intrinsics.model in {"simple_pinhole", "pinhole"}:
        # The half-space a perspective divide is defined on.
        return points_camera[:, 2] > PERSPECTIVE_DEPTH_FLOOR
    if intrinsics.model == "ortho":
        # Parallel rays reach the whole depth axis, leaving an ortho camera no behind.
        return torch.ones(
            (points_camera.shape[0],),
            device=points_camera.device,
            dtype=torch.bool,
        )
    assert 0, "Should not reach here. " f"{intrinsics.model=}"


def compute_camera_view_directions(
    points_camera: torch.Tensor,
    intrinsics: CameraIntrinsics,
) -> torch.Tensor:
    """Compute each camera-space point's unit direction back toward the camera.

    Args:
        points_camera: Camera-space points [N, 3].
        intrinsics: The view's CameraIntrinsics, whose model decides how the rays run.

    Returns:
        Unit view directions [N, 3], each running from its point back toward the
        camera.
    """

    def _validate_inputs() -> None:
        # Input validations
        assert isinstance(points_camera, torch.Tensor), f"{type(points_camera)=}"
        assert isinstance(intrinsics, CameraIntrinsics), f"{type(intrinsics)=}"
        assert points_camera.ndim == 2, f"{points_camera.shape=}"
        assert points_camera.shape[1] == 3, f"{points_camera.shape=}"

    _validate_inputs()

    if intrinsics.model in {"simple_pinhole", "pinhole"}:
        # Each pinhole ray runs back to the camera's own centre.
        return torch.nn.functional.normalize(-points_camera, dim=1)
    if intrinsics.model == "ortho":
        # The rays run parallel, so one axis stands for every one of them.
        view_direction = torch.zeros_like(points_camera)
        view_direction[:, 2] = -1.0
        return view_direction
    assert 0, "Should not reach here. " f"{intrinsics.model=}"


def _camera_verts_to_clip(
    verts_camera: torch.Tensor,
    intrinsics: CameraIntrinsics,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Convert camera-space verts to clip-space for rasterization.

    Args:
        verts_camera: Camera-space verts [V, 3].
        intrinsics: The view's CameraIntrinsics, whose own model owns the projection.
        image_height: Render height in pixels.
        image_width: Render width in pixels.

    Returns:
        Clip-space verts [1, V, 4].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(verts_camera, torch.Tensor), f"{type(verts_camera)=}"
        assert isinstance(intrinsics, CameraIntrinsics), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert verts_camera.ndim == 2, f"{verts_camera.shape=}"
        assert verts_camera.shape[1] == 3, f"{verts_camera.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    z_camera = verts_camera[:, 2]
    if intrinsics.model in {"simple_pinhole", "pinhole"}:
        # The w a perspective divide would otherwise take to zero.
        z_camera = z_camera.clamp(min=PERSPECTIVE_DEPTH_FLOOR)
        w = z_camera
    elif intrinsics.model == "ortho":
        # An ortho projection divides by no depth, so the homogeneous coordinate is unit.
        w = torch.ones_like(z_camera)
    else:
        assert 0, "Should not reach here. " f"{intrinsics.model=}"

    verts_camera_floored = torch.stack(
        [verts_camera[:, 0], verts_camera[:, 1], z_camera],
        dim=1,
    )
    pixels = intrinsics.project(points_camera=verts_camera_floored, inplace=False)
    x_ndc = (pixels[:, 0] / float(max(image_width - 1, 1))) * 2.0 - 1.0
    y_ndc = 1.0 - (pixels[:, 1] / float(max(image_height - 1, 1))) * 2.0

    z_min = torch.min(z_camera)
    z_max = torch.max(z_camera)
    z_ndc = ((z_camera - z_min) / (z_max - z_min + 1e-6)) * 2.0 - 1.0
    return torch.stack(
        [x_ndc * w, y_ndc * w, z_ndc * w, w],
        dim=1,
    ).unsqueeze(0)
