"""Shared camera-space geometry helpers for mesh texture extraction."""

import nvdiffrast.torch as dr
import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.point_cloud.camera.transform import (
    world_to_camera_transform,
)


def _render_camera_face_index_buffer(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Render a one-view camera-space face-index buffer.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
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
        assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
        assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    clip_vertices = _camera_vertices_to_clip(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=vertices_camera.device, dtype=torch.float32)
    tri_i32 = faces.to(device=vertices_camera.device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=vertices_camera.device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_vertices.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )

    face_indices = rast_out[..., 3].to(dtype=torch.long) - 1
    face_plus1 = (face_indices + 1).to(dtype=torch.float32).unsqueeze(-1)
    visible = face_indices >= 0
    return torch.where(visible.unsqueeze(-1), face_plus1, torch.zeros_like(face_plus1))


def _render_camera_depth_buffer(
    vertices_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Render a one-view camera-space depth buffer.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
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
        assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
        assert isinstance(faces, torch.Tensor), f"{type(faces)=}"
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
        assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    clip_vertices = _camera_vertices_to_clip(
        vertices_camera=vertices_camera,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    ).to(device=vertices_camera.device, dtype=torch.float32)
    tri_i32 = faces.to(device=vertices_camera.device, dtype=torch.int32).contiguous()
    raster_context = dr.RasterizeCudaContext(device=vertices_camera.device)
    rast_out, _ = dr.rasterize(
        glctx=raster_context,
        pos=clip_vertices.contiguous(),
        tri=tri_i32,
        resolution=[image_height, image_width],
        ranges=None,
    )
    depth_image, _ = dr.interpolate(
        attr=vertices_camera[:, 2:3].unsqueeze(0).contiguous(),
        rast=rast_out,
        tri=tri_i32,
    )
    visible = rast_out[..., 3] > 0
    return torch.where(
        visible.unsqueeze(-1),
        depth_image,
        torch.zeros_like(depth_image),
    )


def _vertices_world_to_camera(
    vertices: torch.Tensor,
    camera: Cameras,
) -> torch.Tensor:
    """Transform one-view world-space vertices to camera-space vertices.

    Args:
        vertices: Mesh vertices in world coordinates [V, 3].
        camera: One camera instance.

    Returns:
        Camera-space vertices [V, 3].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert vertices.ndim == 2, f"{vertices.shape=}"
        assert vertices.shape[1] == 3, f"{vertices.shape=}"
        assert len(camera) == 1, f"{len(camera)=}"

    _validate_inputs()

    camera_single = camera[0].to(device=vertices.device, convention="opencv")
    vertices_camera = world_to_camera_transform(
        points=vertices,
        extrinsics=camera_single.extrinsics,
        inplace=False,
    )
    assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
    assert (
        vertices_camera.shape == vertices.shape
    ), f"{vertices_camera.shape=} {vertices.shape=}"
    return vertices_camera


def _project_vertices_to_image(
    vertices: torch.Tensor,
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project world-space vertices to image pixels for one view.

    Args:
        vertices: Mesh vertices in world coordinates [V, 3].
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        A tuple of:
            xy: Pixel coordinates [V, 2].
            depth: Camera-space depth [V].
            vertices_camera: Camera-space vertices [V, 3].
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
        assert isinstance(vertices, torch.Tensor), f"{type(vertices)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert vertices.ndim == 2, f"{vertices.shape=}"
        assert vertices.shape[1] == 3, f"{vertices.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"
        assert len(camera) == 1, f"{len(camera)=}"

    _validate_inputs()

    camera_single = camera[0].to(device=vertices.device, convention="opencv")
    intrinsics = camera_single.intrinsics
    vertices_camera = _vertices_world_to_camera(
        vertices=vertices,
        camera=camera,
    )
    depth = vertices_camera[:, 2]

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    x = fx * (vertices_camera[:, 0] / depth) + cx
    y = fy * (vertices_camera[:, 1] / depth) + cy

    valid = (
        (depth > 1e-8)
        & (x >= 0.0)
        & (x <= float(image_width - 1))
        & (y >= 0.0)
        & (y <= float(image_height - 1))
    )
    xy = torch.stack([x, y], dim=1)
    return xy, depth, vertices_camera, valid


def _camera_vertices_to_clip(
    vertices_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Convert camera-space vertices to clip-space for rasterization.

    Args:
        vertices_camera: Camera-space vertices [V, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Render height in pixels.
        image_width: Render width in pixels.

    Returns:
        Clip-space vertices [1, V, 4].
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(vertices_camera, torch.Tensor), f"{type(vertices_camera)=}"
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert vertices_camera.ndim == 2, f"{vertices_camera.shape=}"
        assert vertices_camera.shape[1] == 3, f"{vertices_camera.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    x_camera = vertices_camera[:, 0]
    y_camera = vertices_camera[:, 1]
    z_camera = vertices_camera[:, 2].clamp(min=1e-6)

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    x_pixel = fx * (x_camera / z_camera) + cx
    y_pixel = fy * (y_camera / z_camera) + cy
    x_ndc = (x_pixel / float(max(image_width - 1, 1))) * 2.0 - 1.0
    y_ndc = 1.0 - (y_pixel / float(max(image_height - 1, 1))) * 2.0

    z_min = torch.min(z_camera)
    z_max = torch.max(z_camera)
    z_ndc = ((z_camera - z_min) / (z_max - z_min + 1e-6)) * 2.0 - 1.0
    w = z_camera
    return torch.stack(
        [x_ndc * w, y_ndc * w, z_ndc * w, w],
        dim=1,
    ).unsqueeze(0)
