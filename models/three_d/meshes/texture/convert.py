"""
Texture representation conversion utilities for generic triangle meshes.
"""

from typing import Tuple

import nvdiffrast.torch as dr
import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def _verts_uvs_to_clip(
    verts_uvs: torch.Tensor,
) -> torch.Tensor:
    """Convert UV coordinates to rasterization clip-space coordinates.

    Args:
        verts_uvs: UV-coordinate tensor with shape ``[V, 2]``.

    Returns:
        Homogeneous clip-space positions with shape ``[1, V, 4]``.
    """

    x = verts_uvs[:, 0] * 2.0 - 1.0
    y = 1.0 - verts_uvs[:, 1] * 2.0
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
        mesh: Mesh providing ``faces``, ``device``, and a
            ``MeshTextureUVTextureMap`` texture supplying ``verts_uvs``.
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
        assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Expected `mesh` to carry a `MeshTextureUVTextureMap` texture. "
            f"{type(mesh.texture)=}"
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
        vertex_count = mesh.texture.verts_uvs.shape[0]
        if vertex_feature.ndim == 2:
            assert vertex_feature.shape[0] == vertex_count, (
                "Expected `vertex_feature` to align with `mesh.texture.verts_uvs`. "
                f"{vertex_feature.shape=} {mesh.texture.verts_uvs.shape=}"
            )
        else:
            assert vertex_feature.shape[0] == 1, (
                "Expected batched `vertex_feature` to contain one mesh. "
                f"{vertex_feature.shape=}"
            )
            assert vertex_feature.shape[1] == vertex_count, (
                "Expected `vertex_feature` to align with `mesh.texture.verts_uvs`. "
                f"{vertex_feature.shape=} {mesh.texture.verts_uvs.shape=}"
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

    uv_clip = _verts_uvs_to_clip(verts_uvs=mesh.texture.verts_uvs).to(
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
    vertex_colored_mesh: Mesh,
    uv_layout: MeshTextureUVTextureMap,
    texture_size: int,
) -> MeshTextureUVTextureMap:
    """Bake per-vertex colors onto a UV texture map via rasterization.

    The vertex colors and the UV layout are two separate ``MeshTexture``
    objects over the same ``verts`` / ``faces``: one mesh carries the
    vertex-color texture, and ``uv_layout`` carries the UV coordinates and
    UV-face indices. The ``uv_layout`` UV convention is converted internally;
    callers do not need to pre-convert.

    Args:
        vertex_colored_mesh: Mesh carrying a ``MeshTextureVertexColor`` texture
            and providing ``faces`` and ``device``.
        uv_layout: UV layout texture over the same ``verts`` / ``faces``,
            supplying ``verts_uvs``, ``faces_uvs``, and ``convention``. Its
            ``uv_texture_map`` image is ignored; only the UV layout is used.
        texture_size: Output square texture resolution.

    Returns:
        Baked ``MeshTextureUVTextureMap`` whose ``uv_texture_map`` holds the
        rasterized per-vertex colors in HWC float32 ``[0, 1]`` and whose
        ``verts_uvs`` / ``faces_uvs`` / ``convention`` match ``uv_layout``.
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_colored_mesh, Mesh), (
            "Expected `vertex_colored_mesh` to be a `Mesh`. "
            f"{type(vertex_colored_mesh)=}"
        )
        assert isinstance(vertex_colored_mesh.texture, MeshTextureVertexColor), (
            "Expected `vertex_colored_mesh` to carry a `MeshTextureVertexColor` "
            "texture. "
            f"{type(vertex_colored_mesh.texture)=}"
        )
        assert isinstance(uv_layout, MeshTextureUVTextureMap), (
            "Expected `uv_layout` to be a `MeshTextureUVTextureMap`. "
            f"{type(uv_layout)=}"
        )
        assert (
            vertex_colored_mesh.texture.vertex_color.shape[0]
            == uv_layout.verts_uvs.shape[0]
        ), (
            "Expected the vertex colors to align with `uv_layout.verts_uvs` "
            "because this function uses `faces` as the shared index buffer. "
            f"{vertex_colored_mesh.texture.vertex_color.shape=} "
            f"{uv_layout.verts_uvs.shape=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> MeshTextureUVTextureMap:
        device = vertex_colored_mesh.device
        return uv_layout.to(device=device, verts_uvs_convention="obj")

    uv_layout = _normalize_inputs()

    vertex_feature = vertex_colored_mesh.texture.vertex_color.to(
        device=vertex_colored_mesh.device
    ).unsqueeze(0)

    rasterization_mesh = Mesh(
        verts=vertex_colored_mesh.verts,
        faces=vertex_colored_mesh.faces,
        texture=uv_layout,
    )
    texel_color, mask = rasterize_vertex_features_to_uv_map(
        mesh=rasterization_mesh,
        vertex_feature=vertex_feature,
        texture_size=texture_size,
    )
    mean_color = vertex_feature.mean(dim=1, keepdim=True).unsqueeze(1)
    uv_texture_map = (texel_color * mask + mean_color * (1.0 - mask)).clamp(
        min=0.0, max=1.0
    )
    return MeshTextureUVTextureMap(
        uv_texture_map=uv_texture_map.contiguous(),
        verts_uvs=uv_layout.verts_uvs,
        faces_uvs=uv_layout.faces_uvs,
        convention=uv_layout.convention,
    )
