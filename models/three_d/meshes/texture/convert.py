"""
Texture representation conversion utilities for generic triangle meshes.
"""

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import nvdiffrast.torch as dr
import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.validate import validate_uv_texture_map

REPO_ROOT = Path(__file__).resolve().parents[4]
CANONICAL_BFM_VERTEX_UV_PATH = (
    REPO_ROOT
    / "project"
    / "submodules"
    / "HRN"
    / "assets"
    / "3dmm_assets"
    / "template_mesh"
    / "bfm_uvs2.npy"
)


@lru_cache(maxsize=1)
def _load_canonical_bfm_vertex_uv_cpu() -> torch.Tensor:
    """Load the topology-compatible canonical BFM UV table on CPU.

    Args:
        None.

    Returns:
        Canonical BFM UV tensor with shape `[V, 2]` on CPU.
    """

    assert CANONICAL_BFM_VERTEX_UV_PATH.exists(), (
        "Expected the canonical BFM UV asset to exist. "
        f"{CANONICAL_BFM_VERTEX_UV_PATH=}"
    )
    canonical_bfm_vertex_uv = np.load(CANONICAL_BFM_VERTEX_UV_PATH)
    assert isinstance(canonical_bfm_vertex_uv, np.ndarray), (
        "Expected the canonical BFM UV asset to load as a `numpy.ndarray`. "
        f"{type(canonical_bfm_vertex_uv)=}"
    )
    assert canonical_bfm_vertex_uv.ndim == 2, (
        "Expected the canonical BFM UV asset to have shape `[V, 2]`. "
        f"{canonical_bfm_vertex_uv.shape=}"
    )
    assert canonical_bfm_vertex_uv.shape[1] == 2, (
        "Expected the canonical BFM UV asset to have shape `[V, 2]`. "
        f"{canonical_bfm_vertex_uv.shape=}"
    )
    assert canonical_bfm_vertex_uv.dtype in (np.float32, np.float64), (
        "Expected the canonical BFM UV asset dtype to be float32 or float64. "
        f"{canonical_bfm_vertex_uv.dtype=}"
    )
    canonical_bfm_vertex_uv = torch.from_numpy(canonical_bfm_vertex_uv)
    assert isinstance(canonical_bfm_vertex_uv, torch.Tensor), (
        "Expected the canonical BFM UV asset to convert to a `torch.Tensor`. "
        f"{type(canonical_bfm_vertex_uv)=}"
    )
    assert canonical_bfm_vertex_uv.dtype in (torch.float32, torch.float64), (
        "Expected the canonical BFM UV tensor dtype to be float32 or float64 "
        "before normalization. "
        f"{canonical_bfm_vertex_uv.dtype=}"
    )
    canonical_bfm_vertex_uv = canonical_bfm_vertex_uv.to(dtype=torch.float32)
    assert torch.all(canonical_bfm_vertex_uv >= 0.0), (
        "Expected canonical BFM UV coordinates to stay within `[0, 1]`. "
        f"{float(canonical_bfm_vertex_uv.min())=}"
    )
    assert torch.all(canonical_bfm_vertex_uv <= 1.0), (
        "Expected canonical BFM UV coordinates to stay within `[0, 1]`. "
        f"{float(canonical_bfm_vertex_uv.max())=}"
    )
    return canonical_bfm_vertex_uv.contiguous()


def build_canonical_bfm_vertex_uv(
    mean_shape: torch.Tensor,
) -> torch.Tensor:
    """Build the canonical BFM UV layout for one BFM-topology mesh.

    Args:
        mean_shape: Flattened or `[V, 3]` BFM-shape tensor used for topology/device
            validation.

    Returns:
        Canonical BFM UV tensor with shape `[V, 2]`.
    """

    def _validate_inputs() -> None:
        assert isinstance(mean_shape, torch.Tensor), (
            "Expected `mean_shape` to be a `torch.Tensor`. " f"{type(mean_shape)=}"
        )
        assert mean_shape.ndim in (1, 2), (
            "Expected `mean_shape` to be flat or `[V, 3]`. " f"{mean_shape.shape=}"
        )
        assert mean_shape.numel() % 3 == 0, (
            "Expected `mean_shape` to contain xyz triplets. " f"{mean_shape.shape=}"
        )

    _validate_inputs()

    vertex_count = int(mean_shape.reshape(-1, 3).shape[0])
    canonical_bfm_vertex_uv = _load_canonical_bfm_vertex_uv_cpu()
    assert canonical_bfm_vertex_uv.shape == (vertex_count, 2), (
        "Expected the canonical BFM UV table to match the requested vertex "
        "count. "
        f"{canonical_bfm_vertex_uv.shape=} {vertex_count=}"
    )
    return canonical_bfm_vertex_uv.to(
        device=mean_shape.device,
        dtype=torch.float32,
    ).contiguous()


def _vertex_uv_to_clip(
    vertex_uv: torch.Tensor,
) -> torch.Tensor:
    """Convert UV coordinates to rasterization clip-space coordinates.

    Args:
        vertex_uv: UV-coordinate tensor with shape ``[V, 2]``.

    Returns:
        Homogeneous clip-space positions with shape ``[1, V, 4]``.
    """

    x = vertex_uv[:, 0] * 2.0 - 1.0
    y = 1.0 - vertex_uv[:, 1] * 2.0
    z = torch.zeros_like(x)
    w = torch.ones_like(x)
    return torch.stack([x, y, z, w], dim=1).unsqueeze(0)


def rasterize_vertex_features_to_uv_map(
    mesh: Mesh,
    vertex_feature: torch.Tensor,
    texture_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize per-vertex features onto the mesh UV map.

    Args:
        mesh: Mesh providing ``vertex_uv``, ``faces``, and ``device``.
        vertex_feature: Per-vertex feature tensor with shape ``[V, C]`` or
            ``[1, V, C]``.
        texture_size: Output square texture resolution.

    Returns:
        Tuple of rasterized feature map ``[1, texture_size, texture_size, C]`` and
        valid texel mask ``[1, texture_size, texture_size, 1]``.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert mesh.vertex_uv is not None, (
            "Expected `mesh` to carry UV coordinates. " f"{mesh.vertex_uv=}"
        )
        assert isinstance(vertex_feature, torch.Tensor), (
            "Expected `vertex_feature` to be a tensor. " f"{type(vertex_feature)=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )
        assert vertex_feature.ndim in (2, 3), (
            "Expected `vertex_feature` to have shape `[V, C]` or `[1, V, C]`. "
            f"{vertex_feature.shape=}"
        )
        vertex_count = mesh.vertex_uv.shape[0]
        if vertex_feature.ndim == 2:
            assert vertex_feature.shape[0] == vertex_count, (
                "Expected `vertex_feature` to align with `mesh.vertex_uv`. "
                f"{vertex_feature.shape=} {mesh.vertex_uv.shape=}"
            )
        else:
            assert vertex_feature.shape[0] == 1, (
                "Expected batched `vertex_feature` to contain one mesh. "
                f"{vertex_feature.shape=}"
            )
            assert vertex_feature.shape[1] == vertex_count, (
                "Expected `vertex_feature` to align with `mesh.vertex_uv`. "
                f"{vertex_feature.shape=} {mesh.vertex_uv.shape=}"
            )

    _validate_inputs()

    def _normalize_inputs() -> torch.Tensor:
        """Normalize per-vertex features to a one-item batch.

        Args:
            None.

        Returns:
            Per-vertex feature tensor with shape ``[1, V, C]``.
        """

        if vertex_feature.ndim == 2:
            return vertex_feature.unsqueeze(0)
        return vertex_feature

    vertex_feature = _normalize_inputs()

    uv_clip = _vertex_uv_to_clip(vertex_uv=mesh.vertex_uv).to(
        device=mesh.device,
        dtype=torch.float32,
    )
    tri_i32 = mesh.faces.to(device=mesh.device, dtype=torch.int32).contiguous()
    feat = vertex_feature.to(device=mesh.device, dtype=torch.float32).contiguous()

    uv_ctx = dr.RasterizeCudaContext(device=mesh.device)
    rast_out, _ = dr.rasterize(
        glctx=uv_ctx,
        pos=uv_clip.contiguous(),
        tri=tri_i32,
        resolution=[texture_size, texture_size],
        ranges=None,
    )
    texel_feature, _ = dr.interpolate(attr=feat, rast=rast_out, tri=tri_i32)
    mask = (rast_out[..., 3] > 0).float().unsqueeze(-1)
    return texel_feature.contiguous(), mask.contiguous()


def bake_vertex_colors_to_uv_texture_map(
    mesh: Mesh,
    texture_size: int,
) -> torch.Tensor:
    """Bake per-vertex colors onto a UV texture map via rasterization.

    Args:
        mesh: Mesh providing ``vertex_color``, ``vertex_uv``, ``faces``,
            ``convention``, and ``device``.  The UV convention is converted
            internally; callers do not need to pre-convert.
        texture_size: Output square texture resolution.

    Returns:
        UV texture map ``[1, texture_size, texture_size, 3]`` in ``[0, 1]``.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh`. " f"{type(mesh)=}"
        )
        assert mesh.vertex_color is not None, (
            "Expected `mesh` to carry vertex colors. " f"{mesh.vertex_color=}"
        )
        assert mesh.uv_texture_map is None, (
            "Expected `mesh` to not already carry a UV texture map. "
            f"{mesh.uv_texture_map is None=}"
        )
        assert mesh.vertex_uv is not None, (
            "Expected `mesh` to carry UV coordinates. " f"{mesh.vertex_uv=}"
        )
        assert mesh.vertex_color.shape[0] == mesh.vertex_uv.shape[0], (
            "Expected `mesh.vertex_color` to align with `mesh.vertex_uv` because "
            "this function uses `faces` as the shared index buffer. "
            f"{mesh.vertex_color.shape=} {mesh.vertex_uv.shape=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Mesh:
        return mesh.to(convention="obj")

    mesh = _normalize_inputs()

    vertex_color = mesh.vertex_color.unsqueeze(0)

    texel_color, mask = rasterize_vertex_features_to_uv_map(
        mesh=mesh,
        vertex_feature=vertex_color,
        texture_size=texture_size,
    )
    mean_color = vertex_color.mean(dim=1, keepdim=True).unsqueeze(1)
    uv_texture_map = texel_color * mask + mean_color * (1.0 - mask)
    validate_uv_texture_map(obj=uv_texture_map)
    return uv_texture_map.contiguous()
