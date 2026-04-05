"""Shared normal-alignment weighting helpers for texture extraction."""

from typing import Any, Dict

import torch
import torch.nn.functional as F

from data.structures.three_d.camera.cameras import Cameras
from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.ops.normals import compute_vertex_normals
from models.three_d.meshes.texture.extract.camera_geometry import (
    _vertices_world_to_camera,
)
from models.three_d.meshes.texture.extract.weights_cfg import (
    normalize_weights_cfg,
    validate_weights_cfg,
)


def _compute_v_normals_weights(
    mesh: Mesh,
    camera: Cameras,
    weights_cfg: Dict[str, Any],
) -> torch.Tensor:
    """Compute one-view per-vertex normal-alignment weights.

    Args:
        mesh: Extraction mesh.
        camera: One camera instance.
        weights_cfg: One-view weighting configuration dictionary.

    Returns:
        Per-vertex weights `[V]`, computed as clamped normal/view alignment.
    """

    def _validate_inputs() -> None:
        """Validate vertex-normal weighting inputs.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(weights_cfg, dict), f"{type(weights_cfg)=}"
        assert len(camera) == 1, f"{len(camera)=}"
        validate_weights_cfg(weights_cfg=weights_cfg)

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize vertex-normal weighting inputs.

        Args:
            None.

        Returns:
            Normalized weighting config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="normals",
        )

    weights_cfg = _normalize_inputs()
    normals_weight_power = weights_cfg["normals_weight_power"]
    normals_weight_threshold = weights_cfg["normals_weight_threshold"]

    vertices_camera = _vertices_world_to_camera(
        vertices=mesh.vertices,
        camera=camera,
    )
    normals_camera = compute_vertex_normals(
        base_vertices=vertices_camera,
        faces=mesh.faces,
    ).to(device=mesh.device, dtype=torch.float32)
    normals_camera_norm = torch.linalg.norm(normals_camera, dim=1)
    normals_camera_norm_error = torch.max(torch.abs(normals_camera_norm - 1.0))
    assert (
        float(normals_camera_norm_error) <= 1.0e-5
    ), f"{float(normals_camera_norm_error)=}"

    view_direction = F.normalize(-vertices_camera, p=2, dim=1)
    alignment = (normals_camera * view_direction).sum(dim=1).clamp(0.0, 1.0)
    assert torch.all(alignment >= 0.0), f"{float(alignment.min())=}"
    assert torch.all(alignment <= 1.0), f"{float(alignment.max())=}"
    alignment = torch.where(
        alignment >= normals_weight_threshold,
        alignment,
        torch.zeros_like(alignment),
    )
    alignment = alignment.pow(normals_weight_power)
    return alignment


def _compute_f_normals_weights(
    mesh: Mesh,
    camera: Cameras,
    weights_cfg: Dict[str, Any],
) -> torch.Tensor:
    """Compute one-view per-face normal-alignment weights.

    Args:
        mesh: Extraction mesh.
        camera: One camera instance.
        weights_cfg: One-view weighting configuration dictionary.

    Returns:
        Per-face weights `[F]`, computed as clamped normal/view alignment.
    """

    def _validate_inputs() -> None:
        """Validate face-normal weighting inputs.

        Args:
            None.

        Returns:
            None.
        """

        assert isinstance(mesh, Mesh), f"{type(mesh)=}"
        assert isinstance(camera, Cameras), f"{type(camera)=}"
        assert isinstance(weights_cfg, dict), f"{type(weights_cfg)=}"
        assert len(camera) == 1, f"{len(camera)=}"
        validate_weights_cfg(weights_cfg=weights_cfg)

    _validate_inputs()

    def _normalize_inputs() -> Dict[str, Any]:
        """Normalize face-normal weighting inputs.

        Args:
            None.

        Returns:
            Normalized weighting config.
        """

        return normalize_weights_cfg(
            weights_cfg=weights_cfg,
            default_weights="normals",
        )

    weights_cfg = _normalize_inputs()
    normals_weight_power = weights_cfg["normals_weight_power"]
    normals_weight_threshold = weights_cfg["normals_weight_threshold"]

    vertices_camera = _vertices_world_to_camera(
        vertices=mesh.vertices,
        camera=camera,
    )
    v0_camera = vertices_camera[mesh.faces[:, 0]]
    v1_camera = vertices_camera[mesh.faces[:, 1]]
    v2_camera = vertices_camera[mesh.faces[:, 2]]
    face_normals_camera = torch.cross(
        v1_camera - v0_camera,
        v2_camera - v0_camera,
        dim=1,
    )
    face_normals_camera_norm = torch.linalg.norm(
        face_normals_camera,
        dim=1,
        keepdim=True,
    )
    assert torch.all(face_normals_camera_norm > 0), f"{face_normals_camera_norm.min()=}"
    face_normals_camera = face_normals_camera / face_normals_camera_norm

    face_centers_camera = (v0_camera + v1_camera + v2_camera) / 3.0
    face_view_direction = F.normalize(-face_centers_camera, p=2, dim=1)
    alignment = (face_normals_camera * face_view_direction).sum(dim=1).clamp(0.0, 1.0)
    assert torch.all(alignment >= 0.0), f"{float(alignment.min())=}"
    assert torch.all(alignment <= 1.0), f"{float(alignment.max())=}"
    alignment = torch.where(
        alignment >= normals_weight_threshold,
        alignment,
        torch.zeros_like(alignment),
    )
    alignment = alignment.pow(normals_weight_power)
    return alignment
