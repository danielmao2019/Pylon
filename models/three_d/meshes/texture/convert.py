"""
Texture representation conversion utilities for generic triangle meshes.
"""

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import nvdiffrast.torch as dr
import torch

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
        """Validate canonical-BFM UV build inputs.

        Args:
            None.

        Returns:
            None.
        """

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
    # Input validations
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"

    x = vertex_uv[:, 0] * 2.0 - 1.0
    y = 1.0 - vertex_uv[:, 1] * 2.0
    z = torch.zeros_like(x)
    w = torch.ones_like(x)
    return torch.stack([x, y, z, w], dim=1).unsqueeze(0)


def rasterize_vertex_features_to_uv_map(
    vertex_uv: torch.Tensor,
    tri: torch.Tensor,
    vertex_feature: torch.Tensor,
    texture_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Input validations
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(tri, torch.Tensor), f"{type(tri)=}"
    assert isinstance(vertex_feature, torch.Tensor), f"{type(vertex_feature)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert isinstance(device, torch.device), f"{type(device)=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert tri.ndim == 2, f"{tri.shape=}"
    assert tri.shape[1] == 3, f"{tri.shape=}"
    assert vertex_feature.ndim in (2, 3), f"{vertex_feature.shape=}"

    # Input normalizations
    if vertex_feature.ndim == 2:
        vertex_feature = vertex_feature.unsqueeze(0)
    assert vertex_feature.ndim == 3, f"{vertex_feature.shape=}"
    assert vertex_feature.shape[0] == 1, f"{vertex_feature.shape=}"
    assert (
        vertex_feature.shape[1] == vertex_uv.shape[0]
    ), f"{vertex_feature.shape=} {vertex_uv.shape=}"

    uv_clip = _vertex_uv_to_clip(vertex_uv=vertex_uv).to(
        device=device,
        dtype=torch.float32,
    )
    tri_i32 = tri.to(device=device, dtype=torch.int32).contiguous()
    feat = vertex_feature.to(device=device, dtype=torch.float32).contiguous()

    uv_ctx = dr.RasterizeCudaContext(device=device)
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
    vertex_uv: torch.Tensor,
    tri: torch.Tensor,
    vertex_color: torch.Tensor,
    texture_size: int,
    device: torch.device,
) -> torch.Tensor:
    # Input validations
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(tri, torch.Tensor), f"{type(tri)=}"
    assert isinstance(vertex_color, torch.Tensor), f"{type(vertex_color)=}"
    assert isinstance(texture_size, int), f"{type(texture_size)=}"
    assert texture_size > 0, f"{texture_size=}"
    assert isinstance(device, torch.device), f"{type(device)=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert tri.ndim == 2, f"{tri.shape=}"
    assert tri.shape[1] == 3, f"{tri.shape=}"
    assert vertex_color.ndim in (2, 3), f"{vertex_color.shape=}"
    assert vertex_color.shape[-1] == 3, f"{vertex_color.shape=}"

    # Input normalizations
    if vertex_color.ndim == 2:
        vertex_color = vertex_color.unsqueeze(0)
    assert vertex_color.ndim == 3, f"{vertex_color.shape=}"
    assert vertex_color.shape[0] == 1, f"{vertex_color.shape=}"
    assert (
        vertex_color.shape[1] == vertex_uv.shape[0]
    ), f"{vertex_color.shape=} {vertex_uv.shape=}"

    texel_color, mask = rasterize_vertex_features_to_uv_map(
        vertex_uv=vertex_uv,
        tri=tri,
        vertex_feature=vertex_color,
        texture_size=texture_size,
        device=device,
    )
    mean_color = (
        vertex_color.to(device=device, dtype=torch.float32)
        .mean(dim=1, keepdim=True)
        .unsqueeze(1)
    )
    uv_texture_map = texel_color * mask + mean_color * (1.0 - mask)
    return uv_texture_map.clamp(0.0, 1.0).contiguous()
