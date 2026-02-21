"""
Texture representation conversion utilities for generic triangle meshes.
"""

from typing import Tuple

import nvdiffrast.torch as dr
import torch


def build_cylindrical_vertex_uv(
    mean_shape: torch.Tensor,
) -> torch.Tensor:
    # Input validations
    assert isinstance(mean_shape, torch.Tensor), f"{type(mean_shape)=}"
    assert mean_shape.ndim in (1, 2), f"{mean_shape.shape=}"
    assert mean_shape.numel() % 3 == 0, f"{mean_shape.shape=}"

    xyz = mean_shape.reshape(-1, 3)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    u = torch.atan2(x, z) / (2.0 * torch.pi) + 0.5
    y_min = torch.min(y)
    y_max = torch.max(y)
    v = 1.0 - (y - y_min) / (y_max - y_min + 1e-6)
    return torch.stack([u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)], dim=1)


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
