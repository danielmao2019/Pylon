"""Vertex-visibility helpers for mesh texture extraction."""

from typing import Optional

import torch

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.texture.extract.camera_geometry import (
    project_verts_to_image,
    render_camera_face_index_buffer,
)

# -----------------------------------------------------------------------------
# Vertex-visibility API
# -----------------------------------------------------------------------------


def compute_v_visibility_mask(
    mesh: Mesh,
    camera: Cameras,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compute one-view binary visibility mask over verts.

    Args:
        mesh: Mesh with verts `[V, 3]` and faces `[F, 3]`.
        camera: One camera instance.
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Float tensor [V] with values in {0, 1}.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """
        # Input validations
        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert len(camera) == 1, f"{len(camera)=}"
        assert image_height > 0, f"{image_height=}"
        assert image_width > 0, f"{image_width=}"

    _validate_inputs()

    _xy, _depth, verts_camera, projection_valid = project_verts_to_image(
        verts=mesh.verts,
        camera=camera,
        image_height=image_height,
        image_width=image_width,
    )
    visible_vertex_mask = _compute_rasterized_visible_vertex_mask(
        verts_camera=verts_camera,
        faces=mesh.faces.to(device=mesh.device, dtype=torch.long).contiguous(),
        intrinsics=camera[0].intrinsics,
        image_height=image_height,
        image_width=image_width,
    )
    visibility_bool = projection_valid & visible_vertex_mask

    return visibility_bool.to(dtype=torch.float32)


# -----------------------------------------------------------------------------
# Internal vertex-visibility helpers
# -----------------------------------------------------------------------------


def _compute_rasterized_visible_vertex_mask(
    verts_camera: torch.Tensor,
    faces: torch.Tensor,
    intrinsics: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Compute rasterized one-view vertex visibility mask.

    Args:
        verts_camera: Camera-space verts [V, 3].
        faces: Mesh faces [F, 3].
        intrinsics: Camera intrinsics [3, 3].
        image_height: Image height in pixels.
        image_width: Image width in pixels.

    Returns:
        Bool visibility mask over verts [V].
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
        assert isinstance(intrinsics, torch.Tensor), f"{type(intrinsics)=}"
        assert isinstance(image_height, int), f"{type(image_height)=}"
        assert isinstance(image_width, int), f"{type(image_width)=}"
        assert verts_camera.ndim == 2, f"{verts_camera.shape=}"
        assert verts_camera.shape[1] == 3, f"{verts_camera.shape=}"
        assert faces.ndim == 2, f"{faces.shape=}"
        assert faces.shape[1] == 3, f"{faces.shape=}"
        assert intrinsics.shape == (3, 3), f"{intrinsics.shape=}"

    _validate_inputs()

    device = verts_camera.device
    vertex_count = verts_camera.shape[0]
    positive_depth = verts_camera[:, 2] > 1e-8
    if not torch.any(positive_depth):
        return torch.zeros((vertex_count,), device=device, dtype=torch.bool)
    face_front_facing_mask = _compute_face_front_facing_mask(
        verts_camera=verts_camera,
        faces=faces,
    )
    front_facing_faces = faces[face_front_facing_mask].contiguous()
    if front_facing_faces.shape[0] == 0:
        return torch.zeros((vertex_count,), device=device, dtype=torch.bool)

    face_plus1 = render_camera_face_index_buffer(
        verts_camera=verts_camera,
        faces=front_facing_faces,
        intrinsics=intrinsics,
        image_height=image_height,
        image_width=image_width,
    )
    visible_pixels = face_plus1[..., 0] > 0
    visible_vertex_mask = torch.zeros((vertex_count,), device=device, dtype=torch.bool)
    if torch.any(visible_pixels):
        visible_faces = face_plus1[..., 0][visible_pixels].to(dtype=torch.long) - 1
        visible_vertex_indices = front_facing_faces[visible_faces].reshape(-1)
        visible_vertex_mask[visible_vertex_indices.unique()] = True

    return visible_vertex_mask & positive_depth


def _compute_face_front_facing_mask(
    verts_camera: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """Compute which camera-space mesh faces are front-facing.

    Args:
        verts_camera: Camera-space verts [V, 3].
        faces: Mesh faces [F, 3].

    Returns:
        Bool mask [F] where `True` means the face is front-facing.
    """

    def _validate_inputs() -> None:
        """Validate input arguments.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(verts_camera, torch.Tensor), (
            "Expected `verts_camera` to be a tensor. " f"{type(verts_camera)=}"
        )
        assert isinstance(faces, torch.Tensor), (
            "Expected `faces` to be a tensor. " f"{type(faces)=}"
        )
        assert verts_camera.ndim == 2, (
            "Expected `verts_camera` to have shape `[V, 3]`. " f"{verts_camera.shape=}"
        )
        assert verts_camera.shape[1] == 3, (
            "Expected `verts_camera` to have shape `[V, 3]`. " f"{verts_camera.shape=}"
        )
        assert faces.ndim == 2, (
            "Expected `faces` to have shape `[F, 3]`. " f"{faces.shape=}"
        )
        assert faces.shape[1] == 3, (
            "Expected `faces` to have shape `[F, 3]`. " f"{faces.shape=}"
        )

    _validate_inputs()

    v0 = verts_camera[faces[:, 0]]
    v1 = verts_camera[faces[:, 1]]
    v2 = verts_camera[faces[:, 2]]
    face_normals_camera = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals_camera_norm = torch.linalg.norm(
        face_normals_camera,
        dim=1,
        keepdim=True,
    )
    assert torch.all(face_normals_camera_norm > 0.0), (
        "Expected all face normals to have nonzero magnitude. "
        f"{float(face_normals_camera_norm.min())=}"
    )
    face_normals_camera = face_normals_camera / face_normals_camera_norm
    face_centers_camera = (v0 + v1 + v2) / 3.0
    face_view_direction = torch.nn.functional.normalize(
        -face_centers_camera,
        p=2,
        dim=1,
    )
    alignment = (face_normals_camera * face_view_direction).sum(dim=1)
    return alignment > 0.0
