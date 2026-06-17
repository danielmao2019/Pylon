"""Texel -> mesh-face correspondence for UV-textured meshes (nvdiffrast-backed)."""

from typing import TYPE_CHECKING, Dict, List

import nvdiffrast.torch as dr
import torch

from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)

if TYPE_CHECKING:
    from data.structures.three_d.mesh.mesh import Mesh


def build_texel_face_map(
    mesh: "Mesh",
    texture_size: int,
) -> Dict[str, torch.Tensor]:
    """Build the texel -> mesh-face correspondence for one UV-textured mesh (requires a seam-safe canonical MeshTextureUVTextureMap) at the given texture resolution.

    Args:
        mesh: Repo `Mesh` carrying a `MeshTextureUVTextureMap` (seam-safe
            canonical `verts_uvs` and `faces_uvs`).
        texture_size: UV texture resolution `T`.

    Returns:
        Dict with two keys:
            "texel_face_index": `[T, T]` int64 mesh-face index per texel,
                with `-1` sentinel at unoccupied texels.
            "texel_face_barycentric": `[T, T, 3]` float32 barycentric
                weights `(w0, w1, w2)` per texel, summing to 1 on
                occupied texels.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh as MeshClass

        assert isinstance(mesh, MeshClass), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(mesh.texture, MeshTextureUVTextureMap), (
            "Expected `mesh` to carry a `MeshTextureUVTextureMap` texture. "
            f"{type(mesh.texture)=}"
        )
        assert isinstance(texture_size, int), (
            "Expected `texture_size` to be an `int`. " f"{type(texture_size)=}"
        )
        assert texture_size > 0, (
            "Expected `texture_size` to be positive. " f"{texture_size=}"
        )

    _validate_inputs()

    verts_uvs = mesh.texture.verts_uvs
    faces_uvs = mesh.texture.faces_uvs
    faces = mesh.faces

    soup = _build_seam_safe_uv_triangle_soup(
        verts_uvs=verts_uvs,
        faces=faces,
        faces_uvs=faces_uvs,
    )
    raster_verts_uvs = soup["raster_verts_uvs"]
    tri_i32 = soup["tri_i32"]
    raster_face_indices = soup["raster_face_indices"]

    uv_clip = _verts_uvs_to_clip(verts_uvs=raster_verts_uvs).to(
        device=verts_uvs.device, dtype=torch.float32
    )
    uv_context = dr.RasterizeCudaContext(device=verts_uvs.device)
    rast_out, _ = dr.rasterize(
        glctx=uv_context,
        pos=uv_clip.contiguous(),
        tri=tri_i32,
        resolution=[texture_size, texture_size],
        ranges=None,
    )

    texel_face_index = _compute_texel_face_index(
        rast_out=rast_out,
        raster_face_indices=raster_face_indices,
    )
    texel_face_barycentric = _compute_texel_face_barycentric(rast_out=rast_out)

    return {
        "texel_face_index": texel_face_index,
        "texel_face_barycentric": texel_face_barycentric,
    }


def _build_seam_safe_uv_triangle_soup(
    verts_uvs: torch.Tensor,
    faces: torch.Tensor,
    faces_uvs: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build the per-face UV triangle soup, adding a u-shifted mirror copy for any face whose seam-safe corners extend outside [0, 1] so the T x T rasterizer covers both sides of the cylindrical wrap.

    Args:
        verts_uvs: Seam-safe canonical UV-coordinate table `[U, 2]`.
        faces: Mesh faces `[F, 3]`.
        faces_uvs: Face-to-UV index tensor `[F, 3]`.

    Returns:
        Dict with three keys:
            "raster_verts_uvs": Triangle-soup UV coordinates `[Vr, 2]`.
            "tri_i32": Triangle-soup indices `[Fr, 3]` int32.
            "raster_face_indices": Original mesh-face index for each soup
                triangle `[Fr]`.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a `torch.Tensor`. " f"{type(verts_uvs)=}"
        )
        assert verts_uvs.ndim == 2 and verts_uvs.shape[1] == 2, (
            "Expected `verts_uvs` shape `[U, 2]`. " f"{verts_uvs.shape=}"
        )
        assert isinstance(faces, torch.Tensor), (
            "Expected `faces` to be a `torch.Tensor`. " f"{type(faces)=}"
        )
        assert faces.ndim == 2 and faces.shape[1] == 3, (
            "Expected `faces` shape `[F, 3]`. " f"{faces.shape=}"
        )
        assert isinstance(faces_uvs, torch.Tensor), (
            "Expected `faces_uvs` to be a `torch.Tensor`. " f"{type(faces_uvs)=}"
        )
        assert faces_uvs.shape == faces.shape, (
            "Expected `faces_uvs` to align with `faces`. "
            f"{faces_uvs.shape=} {faces.shape=}"
        )

    _validate_inputs()

    faces_uvs_long = faces_uvs.to(
        device=verts_uvs.device, dtype=torch.long
    ).contiguous()
    face_corner_uvs = verts_uvs[faces_uvs_long].to(dtype=torch.float32)
    face_max_u = face_corner_uvs[..., 0].max(dim=1).values
    face_min_u = face_corner_uvs[..., 0].min(dim=1).values
    mirror_minus_one_mask = face_max_u > 1.0
    mirror_plus_one_mask = face_min_u < 0.0
    needs_mirror_mask = mirror_minus_one_mask | mirror_plus_one_mask

    raster_uv_chunks: List[torch.Tensor] = []
    raster_face_index_chunks: List[torch.Tensor] = []

    all_face_indices = torch.arange(
        int(faces.shape[0]),
        dtype=torch.long,
        device=verts_uvs.device,
    )
    raster_uv_chunks.append(face_corner_uvs.reshape(-1, 2).contiguous())
    raster_face_index_chunks.append(all_face_indices)

    if bool(needs_mirror_mask.any().item()):
        mirror_indices = torch.nonzero(needs_mirror_mask, as_tuple=False).reshape(-1)
        mirror_uvs = face_corner_uvs[mirror_indices].clone()
        mirror_minus_one_local = mirror_minus_one_mask[mirror_indices]
        mirror_plus_one_local = mirror_plus_one_mask[mirror_indices]
        mirror_uvs[mirror_minus_one_local, :, 0] = (
            mirror_uvs[mirror_minus_one_local, :, 0] - 1.0
        )
        mirror_uvs[mirror_plus_one_local, :, 0] = (
            mirror_uvs[mirror_plus_one_local, :, 0] + 1.0
        )
        raster_uv_chunks.append(mirror_uvs.reshape(-1, 2).contiguous())
        raster_face_index_chunks.append(mirror_indices)

    raster_verts_uvs = torch.cat(raster_uv_chunks, dim=0).contiguous()
    raster_face_indices = torch.cat(raster_face_index_chunks, dim=0).contiguous()
    tri_i32 = torch.arange(
        start=0,
        end=int(raster_verts_uvs.shape[0]),
        device=verts_uvs.device,
        dtype=torch.int32,
    ).reshape(-1, 3)
    assert int(tri_i32.shape[0]) == int(raster_face_indices.shape[0]), (
        "Expected one soup triangle per emitted face copy. "
        f"{tri_i32.shape=} {raster_face_indices.shape=}"
    )

    return {
        "raster_verts_uvs": raster_verts_uvs,
        "tri_i32": tri_i32,
        "raster_face_indices": raster_face_indices,
    }


def _verts_uvs_to_clip(verts_uvs: torch.Tensor) -> torch.Tensor:
    """Convert UV coordinates to clip-space positions [1, V, 4] for the UV rasterizer (u, v -> 2u - 1, 2v - 1, 0, 1).

    Args:
        verts_uvs: Per-vertex UV coordinates `[V, 2]`.

    Returns:
        Clip-space UV verts `[1, V, 4]`.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a `torch.Tensor`. " f"{type(verts_uvs)=}"
        )
        assert verts_uvs.ndim == 2 and verts_uvs.shape[1] == 2, (
            "Expected `verts_uvs` shape `[V, 2]`. " f"{verts_uvs.shape=}"
        )

    _validate_inputs()

    x = verts_uvs[:, 0] * 2.0 - 1.0
    y = verts_uvs[:, 1] * 2.0 - 1.0
    z = torch.zeros_like(x)
    w = torch.ones_like(x)
    return torch.stack([x, y, z, w], dim=1).unsqueeze(0)


def _compute_texel_face_index(
    rast_out: torch.Tensor,
    raster_face_indices: torch.Tensor,
) -> torch.Tensor:
    """Map the rasterizer's per-texel soup-triangle index back to the original mesh-face index, with -1 at unoccupied texels.

    Args:
        rast_out: nvdiffrast output `[1, T, T, 4]` whose last channel is the
            1-indexed soup-triangle id (0 = empty).
        raster_face_indices: Mesh-face id per soup triangle `[Fr]`.

    Returns:
        `[T, T]` int64 mesh-face index map with `-1` at unoccupied texels.
    """

    def _validate_inputs() -> None:
        assert isinstance(rast_out, torch.Tensor), (
            "Expected `rast_out` to be a `torch.Tensor`. " f"{type(rast_out)=}"
        )
        assert (
            rast_out.ndim == 4 and rast_out.shape[0] == 1 and rast_out.shape[3] == 4
        ), ("Expected `rast_out` shape `[1, T, T, 4]`. " f"{rast_out.shape=}")
        assert isinstance(raster_face_indices, torch.Tensor), (
            "Expected `raster_face_indices` to be a `torch.Tensor`. "
            f"{type(raster_face_indices)=}"
        )
        assert raster_face_indices.ndim == 1, (
            "Expected `raster_face_indices` shape `[Fr]`. "
            f"{raster_face_indices.shape=}"
        )

    _validate_inputs()

    soup_triangle_index = rast_out[0, :, :, 3].to(dtype=torch.long) - 1
    occupied_mask = soup_triangle_index >= 0
    texel_face_index = torch.full_like(soup_triangle_index, -1, dtype=torch.int64)
    raster_face_indices_long = raster_face_indices.to(
        device=texel_face_index.device, dtype=torch.int64
    )
    texel_face_index[occupied_mask] = raster_face_indices_long[
        soup_triangle_index[occupied_mask]
    ]
    return texel_face_index.contiguous()


def _compute_texel_face_barycentric(rast_out: torch.Tensor) -> torch.Tensor:
    """Extract per-texel face-local barycentric weights (summing to 1 on occupied texels) from the rasterizer output.

    Args:
        rast_out: nvdiffrast output `[1, T, T, 4]` whose first two channels
            are `(u_bary, v_bary)` within the rasterized soup triangle.

    Returns:
        `[T, T, 3]` float32 barycentric weights `(w0, w1, w2) = (u_bary,
        v_bary, 1 - u_bary - v_bary)`, where `w_k` is the weight of triangle
        corner `k` (nvdiffrast's `u_bary`/`v_bary` are the weights of corners 0
        and 1; corner 2 takes the remainder).
    """

    def _validate_inputs() -> None:
        assert isinstance(rast_out, torch.Tensor), (
            "Expected `rast_out` to be a `torch.Tensor`. " f"{type(rast_out)=}"
        )
        assert (
            rast_out.ndim == 4 and rast_out.shape[0] == 1 and rast_out.shape[3] == 4
        ), ("Expected `rast_out` shape `[1, T, T, 4]`. " f"{rast_out.shape=}")

    _validate_inputs()

    u_bary = rast_out[0, :, :, 0]
    v_bary = rast_out[0, :, :, 1]
    w2 = 1.0 - u_bary - v_bary
    return torch.stack([u_bary, v_bary, w2], dim=-1).contiguous()
