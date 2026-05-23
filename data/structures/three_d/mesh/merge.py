"""Mesh merge helpers owned by the data-layer mesh package."""

from typing import Dict, List, Sequence, Tuple

import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def merge_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh:
    """Merge one or more mesh blocks into one Mesh.

    Args:
        mesh_blocks: Mesh blocks discovered under one mesh root.

    Returns:
        One merged `Mesh`.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh_blocks, (list, tuple)), (
            "Expected `mesh_blocks` to be a list or tuple. " f"{type(mesh_blocks)=}"
        )
        assert len(mesh_blocks) > 0, (
            "Expected at least one mesh block to merge. " f"{len(mesh_blocks)=}"
        )
        assert all(isinstance(mesh, Mesh) for mesh in mesh_blocks), (
            "Expected every mesh block to be a `Mesh` instance. "
            f"{[type(mesh) for mesh in mesh_blocks]=}"
        )
        has_uv_texture_map = any(
            isinstance(mesh.texture, MeshTextureUVTextureMap) for mesh in mesh_blocks
        )
        has_vertex_color = any(
            isinstance(mesh.texture, MeshTextureVertexColor) for mesh in mesh_blocks
        )
        assert not (has_uv_texture_map and has_vertex_color), (
            "Expected mesh blocks to keep a single texture representation. "
            f"{has_uv_texture_map=} {has_vertex_color=}"
        )

    _validate_inputs()

    if len(mesh_blocks) == 1:
        return mesh_blocks[0]

    if any(isinstance(mesh.texture, MeshTextureUVTextureMap) for mesh in mesh_blocks):
        return _merge_uv_textured_meshes(mesh_blocks=mesh_blocks)
    if any(isinstance(mesh.texture, MeshTextureVertexColor) for mesh in mesh_blocks):
        return _merge_vertex_color_meshes(mesh_blocks=mesh_blocks)
    return _merge_geometry_only_meshes(mesh_blocks=mesh_blocks)


def _merge_vertex_color_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh:
    """Merge multiple vertex-colored meshes into one mesh.

    Args:
        mesh_blocks: Vertex-colored mesh blocks (already validated as `Mesh`
            instances by `merge_meshes`).

    Returns:
        One merged vertex-colored `Mesh`.
    """

    def _validate_inputs() -> None:
        assert all(
            isinstance(mesh.texture, MeshTextureVertexColor) for mesh in mesh_blocks
        ), (
            "Expected every block of a vertex-color merge to carry a "
            "`MeshTextureVertexColor`. "
            f"{[type(mesh.texture) for mesh in mesh_blocks]=}"
        )

    _validate_inputs()

    verts_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    vertex_color_list: List[torch.Tensor] = []
    vertex_offset = 0
    for mesh in mesh_blocks:
        verts_list.append(mesh.verts)
        faces_list.append(mesh.faces + vertex_offset)
        vertex_color_list.append(mesh.texture.vertex_color)
        vertex_offset += int(mesh.verts.shape[0])

    return Mesh(
        verts=torch.cat(verts_list, dim=0),
        faces=torch.cat(faces_list, dim=0),
        texture=MeshTextureVertexColor(
            vertex_color=torch.cat(vertex_color_list, dim=0)
        ),
    )


def _merge_uv_textured_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh:
    """Merge multiple UV-textured meshes into one packed textured mesh.

    Args:
        mesh_blocks: UV-textured mesh blocks (already validated as `Mesh`
            instances by `merge_meshes`).

    Returns:
        One merged UV-textured `Mesh`.
    """

    def _validate_inputs() -> None:
        assert all(
            isinstance(mesh.texture, MeshTextureUVTextureMap) for mesh in mesh_blocks
        ), (
            "Expected every block of a UV-textured merge to carry a "
            "`MeshTextureUVTextureMap`. "
            f"{[type(mesh.texture) for mesh in mesh_blocks]=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> List[Mesh]:
        return [mesh.to(convention="obj") for mesh in mesh_blocks]

    mesh_blocks = _normalize_inputs()

    verts_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    verts_uvs_list: List[torch.Tensor] = []
    faces_uvs_list: List[torch.Tensor] = []
    texture_maps: List[torch.Tensor] = []
    face_materials_list: List[torch.Tensor] = []
    vertex_offset = 0
    uv_offset = 0
    for block_index, mesh in enumerate(mesh_blocks):
        verts_list.append(mesh.verts)
        faces_list.append(mesh.faces + vertex_offset)
        verts_uvs_list.append(mesh.texture.verts_uvs)
        faces_uvs_list.append(mesh.texture.faces_uvs + uv_offset)
        texture_maps.append(mesh.texture.uv_texture_map)
        face_materials_list.append(
            torch.full(
                (mesh.faces.shape[0],),
                block_index,
                device=mesh.faces.device,
                dtype=torch.long,
            )
        )
        vertex_offset += int(mesh.verts.shape[0])
        uv_offset += int(mesh.texture.verts_uvs.shape[0])

    packed_texture_map, merged_verts_uvs, merged_faces_uvs = _pack_texture_maps(
        texture_maps=texture_maps,
        verts_uvs=torch.cat(verts_uvs_list, dim=0),
        faces_uvs=torch.cat(faces_uvs_list, dim=0),
        materials_idx=torch.cat(face_materials_list, dim=0),
    )

    return Mesh(
        verts=torch.cat(verts_list, dim=0),
        faces=torch.cat(faces_list, dim=0),
        texture=MeshTextureUVTextureMap(
            uv_texture_map=packed_texture_map,
            verts_uvs=merged_verts_uvs,
            faces_uvs=merged_faces_uvs,
            convention="obj",
        ),
    )


def _merge_geometry_only_meshes(mesh_blocks: Sequence[Mesh]) -> Mesh:
    """Merge multiple geometry-only meshes into one mesh.

    Args:
        mesh_blocks: Geometry-only mesh blocks (already validated as `Mesh`
            instances by `merge_meshes`).

    Returns:
        One merged geometry-only `Mesh`.
    """

    def _validate_inputs() -> None:
        assert all(mesh.texture is None for mesh in mesh_blocks), (
            "Expected every block of a geometry-only merge to carry no texture. "
            f"{[type(mesh.texture) for mesh in mesh_blocks]=}"
        )

    _validate_inputs()

    verts_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    vertex_offset = 0
    for mesh in mesh_blocks:
        verts_list.append(mesh.verts)
        faces_list.append(mesh.faces + vertex_offset)
        vertex_offset += int(mesh.verts.shape[0])

    return Mesh(
        verts=torch.cat(verts_list, dim=0),
        faces=torch.cat(faces_list, dim=0),
        texture=None,
    )


def pack_texture_images(
    texture_images: Dict[str, torch.Tensor],
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    materials_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one or more material texture images into one atlas plus UVs.

    Args:
        texture_images: Material-name to texture-image mapping.
        verts_uvs: UV-coordinate table `[U, 2]`.
        faces_uvs: UV-face indices `[F, 3]`.
        materials_idx: Per-face material ids `[F]`.

    Returns:
        Packed texture map, remapped UV coordinates, and remapped UV-face
        indices.
    """

    def _validate_inputs() -> None:
        assert isinstance(texture_images, dict), (
            "Expected `texture_images` to be a dict. " f"{type(texture_images)=}"
        )
        assert len(texture_images) > 0, (
            "Expected at least one material texture image. " f"{len(texture_images)=}"
        )
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a tensor. " f"{type(verts_uvs)=}"
        )
        assert isinstance(faces_uvs, torch.Tensor), (
            "Expected `faces_uvs` to be a tensor. " f"{type(faces_uvs)=}"
        )
        assert isinstance(materials_idx, torch.Tensor), (
            "Expected `materials_idx` to be a tensor. " f"{type(materials_idx)=}"
        )

    _validate_inputs()

    return _pack_texture_maps(
        texture_maps=[texture_images[name] for name in texture_images],
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        materials_idx=materials_idx,
    )


def _pack_texture_maps(
    texture_maps: Sequence[torch.Tensor],
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    materials_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack texture maps into one atlas and rebuild the per-corner UV table.

    Args:
        texture_maps: Texture map tensors in HWC layout.
        verts_uvs: UV-coordinate table `[U, 2]`.
        faces_uvs: UV-face indices `[F, 3]`.
        materials_idx: Per-face material ids `[F]`.

    Returns:
        Packed texture map, remapped UV coordinates, and remapped UV-face
        indices.
    """

    def _validate_inputs() -> None:
        assert isinstance(texture_maps, (list, tuple)), (
            "Expected `texture_maps` to be a list or tuple. " f"{type(texture_maps)=}"
        )
        assert len(texture_maps) > 0, (
            "Expected at least one texture map. " f"{len(texture_maps)=}"
        )
        assert all(
            isinstance(texture_map, torch.Tensor) for texture_map in texture_maps
        ), (
            "Expected every texture map to be a tensor. "
            f"{[type(texture_map) for texture_map in texture_maps]=}"
        )
        assert all(
            texture_map.ndim == 3 and texture_map.shape[2] >= 3
            for texture_map in texture_maps
        ), (
            "Expected every texture map to be HWC with at least 3 channels. "
            f"{[tuple(texture_map.shape) for texture_map in texture_maps]=}"
        )
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a tensor. " f"{type(verts_uvs)=}"
        )
        assert isinstance(faces_uvs, torch.Tensor), (
            "Expected `faces_uvs` to be a tensor. " f"{type(faces_uvs)=}"
        )
        assert isinstance(materials_idx, torch.Tensor), (
            "Expected `materials_idx` to be a tensor. " f"{type(materials_idx)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> List[torch.Tensor]:
        target_device = verts_uvs.device
        target_dtype = texture_maps[0].dtype
        return [
            texture_map[..., :3].to(device=target_device, dtype=target_dtype)
            for texture_map in texture_maps
        ]

    texture_maps = _normalize_inputs()

    atlas_height = sum(int(texture_map.shape[0]) for texture_map in texture_maps)
    atlas_width = max(int(texture_map.shape[1]) for texture_map in texture_maps)
    assert atlas_height > 0 and atlas_width > 0, (
        "Expected packed texture atlas dimensions to be positive. "
        f"{atlas_height=} {atlas_width=}"
    )

    packed_map = torch.zeros(
        (atlas_height, atlas_width, 3),
        dtype=texture_maps[0].dtype,
        device=verts_uvs.device,
    )
    remap_entries: List[torch.Tensor] = []
    y_offset = 0
    for texture_map in texture_maps:
        texture_height = int(texture_map.shape[0])
        texture_width = int(texture_map.shape[1])
        packed_map[y_offset : y_offset + texture_height, 0:texture_width] = texture_map
        remap_entries.append(
            torch.tensor(
                [
                    float(atlas_height - y_offset - texture_height),
                    float(texture_height),
                    float(texture_width),
                ],
                dtype=torch.float32,
                device=verts_uvs.device,
            )
        )
        y_offset += texture_height

    flat_uv_coords = _remap_uvs(
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        map_offsets=torch.stack(remap_entries, dim=0),
        atlas_height=atlas_height,
        atlas_width=atlas_width,
        materials_idx=materials_idx,
    )
    remapped_faces_uvs = torch.arange(
        flat_uv_coords.shape[0],
        dtype=torch.int64,
        device=verts_uvs.device,
    ).view(-1, 3)
    return packed_map.contiguous(), flat_uv_coords.contiguous(), remapped_faces_uvs


def _remap_uvs(
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    map_offsets: torch.Tensor,
    atlas_height: int,
    atlas_width: int,
    materials_idx: torch.Tensor,
) -> torch.Tensor:
    """Rescale and offset each material's UVs into its packed atlas region.

    Args:
        verts_uvs: UV coordinates to be remapped.
        faces_uvs: Face-to-UV indices.
        map_offsets: Tensor `[K, 3]` holding `(y_offset, height, width)` per
            packed region.
        atlas_height: Height of the packed texture.
        atlas_width: Width of the packed texture.
        materials_idx: Per-face material id `[F]`.

    Returns:
        Remapped UV coordinates aligned to the packed texture.
    """

    def _validate_inputs() -> None:
        assert isinstance(verts_uvs, torch.Tensor), (
            "Expected `verts_uvs` to be a tensor. " f"{type(verts_uvs)=}"
        )
        assert isinstance(faces_uvs, torch.Tensor), (
            "Expected `faces_uvs` to be a tensor. " f"{type(faces_uvs)=}"
        )
        assert isinstance(map_offsets, torch.Tensor), (
            "Expected `map_offsets` to be a tensor. " f"{type(map_offsets)=}"
        )
        assert map_offsets.ndim == 2 and map_offsets.shape[1] == 3, (
            "Expected `map_offsets` to be `[K, 3]`. " f"{map_offsets.shape=}"
        )
        assert isinstance(atlas_height, int) and atlas_height > 0, (
            "Expected `atlas_height` to be a positive int. " f"{atlas_height=}"
        )
        assert isinstance(atlas_width, int) and atlas_width > 0, (
            "Expected `atlas_width` to be a positive int. " f"{atlas_width=}"
        )
        assert isinstance(materials_idx, torch.Tensor), (
            "Expected `materials_idx` to be a tensor. " f"{type(materials_idx)=}"
        )
        assert materials_idx.shape[0] == faces_uvs.shape[0], (
            "Expected one material id per UV face. "
            f"{materials_idx.shape=} {faces_uvs.shape=}"
        )

    _validate_inputs()

    flat_uv_coords = verts_uvs.index_select(dim=0, index=faces_uvs.reshape(-1)).clone()
    flat_material_ids = materials_idx.repeat_interleave(3)
    for material_id in torch.unique(flat_material_ids):
        material_index = int(material_id.item())
        assert material_index < int(map_offsets.shape[0]), (
            "Expected each material id to reference one packed texture region. "
            f"{material_index=} {map_offsets.shape=}"
        )
        material_mask = flat_material_ids == material_id
        offset_y, texture_height, texture_width = map_offsets[material_index]
        remapped_uv = flat_uv_coords[material_mask]
        remapped_uv[:, 0] = (remapped_uv[:, 0] * texture_width) / float(atlas_width)
        remapped_uv[:, 1] = (remapped_uv[:, 1] * texture_height + offset_y) / float(
            atlas_height
        )
        flat_uv_coords[material_mask] = remapped_uv

    return flat_uv_coords.contiguous()
