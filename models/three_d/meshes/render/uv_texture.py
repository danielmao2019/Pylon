"""
UV texture rendering helpers for generic triangle meshes.
"""

from typing import Any, Tuple

import nvdiffrast.torch as dr
import torch


def render_uv_texture_aligned(
    renderer: Any,
    face_vertex: torch.Tensor,
    tri: torch.Tensor,
    vertex_uv: torch.Tensor,
    uv_texture_map: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Input validations
    assert renderer is not None
    assert hasattr(renderer, "ndc_proj"), f"{type(renderer)=}"
    assert hasattr(renderer, "rasterize_size"), f"{type(renderer)=}"
    assert hasattr(renderer, "ctx"), f"{type(renderer)=}"
    assert hasattr(renderer, "use_opengl"), f"{type(renderer)=}"
    assert isinstance(face_vertex, torch.Tensor), f"{type(face_vertex)=}"
    assert isinstance(tri, torch.Tensor), f"{type(tri)=}"
    assert isinstance(vertex_uv, torch.Tensor), f"{type(vertex_uv)=}"
    assert isinstance(uv_texture_map, torch.Tensor), f"{type(uv_texture_map)=}"
    assert face_vertex.ndim == 3, f"{face_vertex.shape=}"
    assert face_vertex.shape[2] == 3, f"{face_vertex.shape=}"
    assert face_vertex.shape[0] == 1, f"{face_vertex.shape=}"
    assert tri.ndim == 2, f"{tri.shape=}"
    assert tri.shape[1] == 3, f"{tri.shape=}"
    assert vertex_uv.ndim == 2, f"{vertex_uv.shape=}"
    assert vertex_uv.shape[1] == 2, f"{vertex_uv.shape=}"
    assert uv_texture_map.ndim == 4, f"{uv_texture_map.shape=}"
    assert uv_texture_map.shape[0] == 1, f"{uv_texture_map.shape=}"
    assert uv_texture_map.shape[3] == 3, f"{uv_texture_map.shape=}"
    assert (
        face_vertex.shape[1] == vertex_uv.shape[0]
    ), f"{face_vertex.shape=} {vertex_uv.shape=}"

    device = face_vertex.device
    ndc_proj = renderer.ndc_proj.to(device)
    vertex = torch.cat(
        [face_vertex, torch.ones((1, face_vertex.shape[1], 1), device=device)], dim=-1
    )
    vertex[..., 1] = -vertex[..., 1]
    vertex_ndc = vertex @ ndc_proj.t()
    tri_i32 = tri.to(device=device, dtype=torch.int32).contiguous()

    if renderer.ctx is None:
        if renderer.use_opengl:
            renderer.ctx = dr.RasterizeGLContext(device=device)
        else:
            renderer.ctx = dr.RasterizeCudaContext(device=device)

    rast_out, _ = dr.rasterize(
        renderer.ctx,
        vertex_ndc.contiguous(),
        tri_i32,
        resolution=[int(renderer.rasterize_size), int(renderer.rasterize_size)],
        ranges=None,
    )
    mask = (rast_out[..., 3] > 0).float().unsqueeze(1)
    uv_feat = vertex_uv.unsqueeze(0).to(device=device, dtype=torch.float32).contiguous()
    uv_interp_internal, _ = dr.interpolate(uv_feat, rast_out, tri_i32)
    uv_interp_internal = uv_interp_internal.clamp(0.0, 1.0).contiguous()
    uv_lookup = uv_interp_internal.clone()
    uv_lookup[..., 1] = 1.0 - uv_lookup[..., 1]
    uv_lookup = uv_lookup.clamp(0.0, 1.0).contiguous()
    uv_tex = uv_texture_map.to(device=device, dtype=torch.float32).contiguous()
    image = dr.texture(uv_tex, uv_lookup, filter_mode="linear")
    image = image.permute(0, 3, 1, 2).contiguous()
    image = image * mask
    return mask, image
