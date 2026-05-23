"""UV texture rendering helpers for generic triangle meshes."""

from typing import Any, Tuple

import nvdiffrast.torch as dr
import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)


def render_uv_texture_aligned(
    renderer: Any,
    mesh: Mesh,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a UV-textured mesh in the renderer's aligned image space.

    The mesh carries a ``MeshTextureUVTextureMap`` texture holding its UV
    convention, ``uv_texture_map``, and UV coordinates, plus camera-space
    ``verts``. This function converts to the convention required by
    ``dr.texture`` internally; callers do not need to pre-convert.

    Args:
        renderer: Deep3DMM renderer exposing ``ndc_proj``, ``rasterize_size``,
            ``ctx``, and ``use_opengl``.
        mesh: Mesh providing camera-space ``verts`` ``[V, 3]``, ``faces``,
            and a ``MeshTextureUVTextureMap`` texture supplying ``verts_uvs``,
            ``convention``, and ``uv_texture_map`` (HWC float32 ``[H, W, 3]``).

    Returns:
        Tuple of (mask ``[1, 1, H, W]``, rendered RGB ``[1, 3, H, W]``).
    """

    def _validate_inputs() -> None:
        assert renderer is not None, "Expected `renderer` to be provided."
        assert hasattr(renderer, "ndc_proj"), (
            "Expected `renderer` to expose `ndc_proj`. " f"{type(renderer)=}"
        )
        assert hasattr(renderer, "rasterize_size"), (
            "Expected `renderer` to expose `rasterize_size`. " f"{type(renderer)=}"
        )
        assert hasattr(renderer, "ctx"), (
            "Expected `renderer` to expose `ctx`. " f"{type(renderer)=}"
        )
        assert hasattr(renderer, "use_opengl"), (
            "Expected `renderer` to expose `use_opengl`. " f"{type(renderer)=}"
        )
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh`. " f"{type(mesh)=}"
        )
        assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Expected `mesh` to carry a UV texture map. " f"{type(mesh.texture)=}"
        )
        assert mesh.verts.shape[0] == mesh.texture.verts_uvs.shape[0], (
            "Expected `mesh.texture.verts_uvs` to align with `mesh.verts` because "
            "this function uses `faces` (not `faces_uvs`) as the shared index buffer "
            "for both rasterization and UV interpolation. "
            f"{mesh.verts.shape=} {mesh.texture.verts_uvs.shape=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Mesh:
        return mesh.to(convention="top_left")

    mesh = _normalize_inputs()

    device = mesh.verts.device
    ndc_proj = renderer.ndc_proj.to(device)
    vertex = torch.cat(
        [
            mesh.verts.unsqueeze(0),
            torch.ones(
                (1, mesh.verts.shape[0], 1),
                device=device,
                dtype=mesh.verts.dtype,
            ),
        ],
        dim=-1,
    )
    vertex[..., 1] = -vertex[..., 1]
    vertex_ndc = vertex @ ndc_proj.t()
    tri_i32 = mesh.faces.to(device=device, dtype=torch.int32).contiguous()

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
    uv_feat = (
        mesh.texture.verts_uvs.unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    uv_interp_internal, _ = dr.interpolate(uv_feat, rast_out, tri_i32)
    uv_lookup = uv_interp_internal.clamp(0.0, 1.0).contiguous()
    uv_tex = (
        mesh.texture.uv_texture_map.unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .contiguous()
    )
    image = dr.texture(uv_tex, uv_lookup, filter_mode="linear")
    image = image.permute(0, 3, 1, 2).contiguous()
    image = image * mask
    return mask, image
