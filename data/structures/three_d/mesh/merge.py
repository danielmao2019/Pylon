"""Mesh merge helpers owned by the data-layer mesh package."""

from typing import TYPE_CHECKING, Dict, List, Sequence, Tuple

import torch

if TYPE_CHECKING:
    from data.structures.three_d.mesh.mesh import Mesh


def merge_meshes(mesh_blocks: Sequence["Mesh"]) -> "Mesh":
    """Merge one or more repo mesh blocks into one repo mesh.

    Args:
        mesh_blocks: Repo mesh blocks discovered under one mesh root.

    Returns:
        One merged repo mesh.
    """

    from data.structures.three_d.mesh.mesh import Mesh

    def _validate_inputs() -> None:
        assert isinstance(mesh_blocks, (list, tuple)), (
            "Expected `mesh_blocks` to be a list or tuple. " f"{type(mesh_blocks)=}"
        )
        assert mesh_blocks, (
            "Expected at least one mesh block to merge. " f"{len(mesh_blocks)=}"
        )
        assert all(isinstance(mesh, Mesh) for mesh in mesh_blocks), (
            "Expected every mesh block to be a `Mesh` instance. " f"{mesh_blocks=}"
        )

    _validate_inputs()

    if len(mesh_blocks) == 1:
        return mesh_blocks[0]

    has_uv_texture_map = any(mesh.uv_texture_map is not None for mesh in mesh_blocks)
    has_vertex_color = any(mesh.vertex_color is not None for mesh in mesh_blocks)

    if has_uv_texture_map:
        assert not has_vertex_color, (
            "Expected textured mesh blocks to keep a single texture representation. "
            f"{has_uv_texture_map=} {has_vertex_color=}"
        )
        return _merge_uv_textured_meshes(mesh_blocks=mesh_blocks)

    if has_vertex_color:
        return _merge_vertex_color_meshes(mesh_blocks=mesh_blocks)

    return _merge_geometry_only_meshes(mesh_blocks=mesh_blocks)


def _merge_vertex_color_meshes(mesh_blocks: Sequence["Mesh"]) -> "Mesh":
    """Merge multiple vertex-colored meshes into one mesh.

    Args:
        mesh_blocks: Vertex-colored repo mesh blocks.

    Returns:
        One merged vertex-colored mesh.
    """

    from data.structures.three_d.mesh.mesh import Mesh

    vertices_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    colors_list: List[torch.Tensor] = []
    vertex_offset = 0
    for mesh in mesh_blocks:
        assert mesh.vertex_color is not None, (
            "Expected vertex-color merge to receive `vertex_color`. "
            f"{mesh.vertex_color is not None=}"
        )
        assert mesh.vertex_uv is None, (
            "Expected vertex-color merge to receive no `vertex_uv`. "
            f"{mesh.vertex_uv is None=}"
        )
        assert mesh.face_uvs is None, (
            "Expected vertex-color merge to receive no `face_uvs`. "
            f"{mesh.face_uvs is None=}"
        )
        assert mesh.convention is None, (
            "Expected vertex-color merge to receive no UV convention. "
            f"{mesh.convention=}"
        )
        vertices_list.append(mesh.vertices)
        faces_list.append(mesh.faces + vertex_offset)
        colors_list.append(mesh.vertex_color)
        vertex_offset += int(mesh.vertices.shape[0])

    return Mesh(
        vertices=torch.cat(vertices_list, dim=0),
        faces=torch.cat(faces_list, dim=0),
        vertex_color=torch.cat(colors_list, dim=0),
    )


def _merge_uv_textured_meshes(mesh_blocks: Sequence["Mesh"]) -> "Mesh":
    """Merge multiple UV-textured meshes into one packed textured mesh.

    Args:
        mesh_blocks: UV-textured repo mesh blocks.

    Returns:
        One merged UV-textured mesh.
    """

    from data.structures.three_d.mesh.mesh import Mesh

    normalized_mesh_blocks = [mesh.to(convention="obj") for mesh in mesh_blocks]
    vertices_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    vertex_uv_list: List[torch.Tensor] = []
    face_uv_list: List[torch.Tensor] = []
    texture_maps: List[torch.Tensor] = []
    face_materials_list: List[torch.Tensor] = []
    vertex_offset = 0
    uv_offset = 0
    for block_index, mesh in enumerate(normalized_mesh_blocks):
        assert mesh.uv_texture_map is not None, (
            "Expected UV-textured merge to receive `uv_texture_map`. "
            f"{mesh.uv_texture_map is not None=}"
        )
        assert mesh.convention == "obj", (
            "Expected UV-textured merge to operate on OBJ-convention UVs. "
            f"{mesh.convention=}"
        )
        vertices_list.append(mesh.vertices)
        faces_list.append(mesh.faces + vertex_offset)
        vertex_uv_list.append(mesh.vertex_uv)
        face_uv_list.append(mesh.face_uvs + uv_offset)
        texture_maps.append(mesh.uv_texture_map)
        face_materials_list.append(
            torch.full(
                (mesh.faces.shape[0],),
                block_index,
                device=mesh.faces.device,
                dtype=torch.long,
            )
        )
        vertex_offset += int(mesh.vertices.shape[0])
        uv_offset += int(mesh.vertex_uv.shape[0])

    merged_vertices = torch.cat(vertices_list, dim=0)
    merged_faces = torch.cat(faces_list, dim=0)
    merged_vertex_uv = torch.cat(vertex_uv_list, dim=0)
    merged_face_uvs = torch.cat(face_uv_list, dim=0)
    merged_face_materials = torch.cat(face_materials_list, dim=0)
    packed_texture_map, merged_vertex_uv, merged_face_uvs = _pack_texture_maps(
        texture_maps=texture_maps,
        verts_uvs=merged_vertex_uv,
        faces_uvs=merged_face_uvs,
        materials_idx=merged_face_materials,
    )

    return Mesh(
        vertices=merged_vertices,
        faces=merged_faces,
        uv_texture_map=packed_texture_map,
        vertex_uv=merged_vertex_uv,
        face_uvs=merged_face_uvs,
        convention="obj",
    )


def _merge_geometry_only_meshes(mesh_blocks: Sequence["Mesh"]) -> "Mesh":
    """Merge multiple geometry-only meshes into one mesh.

    Args:
        mesh_blocks: Geometry-only repo mesh blocks.

    Returns:
        One merged geometry-only mesh.
    """

    from data.structures.three_d.mesh.mesh import Mesh

    vertices_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    vertex_offset = 0
    for mesh in mesh_blocks:
        assert mesh.vertex_color is None, (
            "Expected geometry-only merge to receive no `vertex_color`. "
            f"{mesh.vertex_color is None=}"
        )
        assert mesh.uv_texture_map is None, (
            "Expected geometry-only merge to receive no `uv_texture_map`. "
            f"{mesh.uv_texture_map is None=}"
        )
        assert mesh.vertex_uv is None, (
            "Expected geometry-only merge to receive no `vertex_uv`. "
            f"{mesh.vertex_uv is None=}"
        )
        vertices_list.append(mesh.vertices)
        faces_list.append(mesh.faces + vertex_offset)
        vertex_offset += int(mesh.vertices.shape[0])

    return Mesh(
        vertices=torch.cat(vertices_list, dim=0),
        faces=torch.cat(faces_list, dim=0),
    )


def pack_texture_images(
    texture_images: Dict[str, torch.Tensor],
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    materials_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack one or more texture images into one atlas plus remapped UVs.

    Args:
        texture_images: Material-name to texture-image mapping.
        verts_uvs: UV-coordinate table `[U, 2]`.
        faces_uvs: UV-face indices `[F, 3]`.
        materials_idx: Per-face material ids `[F]`.

    Returns:
        Packed texture map, remapped UV coordinates, and remapped UV-face indices.
    """

    material_names = list(texture_images.keys())
    assert len(material_names) > 0, (
        "Expected textured OBJ loading to provide at least one material texture. "
        f"{material_names=}"
    )
    texture_maps = [texture_images[material_name] for material_name in material_names]
    return _pack_texture_maps(
        texture_maps=texture_maps,
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
    """Pack one or more texture maps into one atlas plus remapped UVs.

    Args:
        texture_maps: Texture map tensors in HWC layout.
        verts_uvs: UV-coordinate table `[U, 2]`.
        faces_uvs: UV-face indices `[F, 3]`.
        materials_idx: Per-face material ids `[F]`.

    Returns:
        Packed texture map, remapped UV coordinates, and remapped UV-face indices.
    """

    def _validate_inputs() -> None:
        assert isinstance(texture_maps, (list, tuple)), (
            "Expected `texture_maps` to be a list or tuple. " f"{type(texture_maps)=}"
        )
        assert len(texture_maps) > 0, (
            "Expected `texture_maps` to contain at least one texture. "
            f"{len(texture_maps)=}"
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

    target_device = verts_uvs.device
    target_dtype = texture_maps[0].dtype
    normalized_texture_maps: List[torch.Tensor] = []
    for texture_tensor in texture_maps:
        assert isinstance(texture_tensor, torch.Tensor), (
            "Expected each texture map to be a tensor. " f"{type(texture_tensor)=}"
        )
        assert texture_tensor.ndim == 3, (
            "Expected each texture map to be rank 3. " f"{texture_tensor.shape=}"
        )
        assert texture_tensor.shape[2] >= 3, (
            "Expected each texture map to have at least 3 channels. "
            f"{texture_tensor.shape=}"
        )
        normalized_texture_maps.append(
            texture_tensor[..., :3].to(device=target_device, dtype=target_dtype)
        )

    atlas_height = sum(
        int(texture_map.shape[0]) for texture_map in normalized_texture_maps
    )
    atlas_width = max(
        int(texture_map.shape[1]) for texture_map in normalized_texture_maps
    )
    assert atlas_height > 0 and atlas_width > 0, (
        "Expected packed texture atlas dimensions to be positive. "
        f"{atlas_height=} {atlas_width=}"
    )

    packed_map = torch.zeros(
        (atlas_height, atlas_width, 3),
        dtype=target_dtype,
        device=target_device,
    )
    remap_entries: List[torch.Tensor] = []
    y_offset = 0
    for texture_tensor in normalized_texture_maps:
        texture_height = int(texture_tensor.shape[0])
        texture_width = int(texture_tensor.shape[1])
        packed_map[y_offset : y_offset + texture_height, 0:texture_width] = (
            texture_tensor
        )
        bottom_offset = float(atlas_height - y_offset - texture_height)
        remap_entries.append(
            torch.tensor(
                [bottom_offset, float(texture_height), float(texture_width)],
                dtype=torch.float32,
                device=target_device,
            )
        )
        y_offset += texture_height

    map_offsets = torch.stack(remap_entries, dim=0)
    flat_uv_coords = _remap_uvs(
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        map_offsets=map_offsets,
        atlas_height=atlas_height,
        atlas_width=atlas_width,
        materials_idx=materials_idx,
    )
    remapped_faces_uvs = torch.arange(
        flat_uv_coords.shape[0],
        dtype=torch.int64,
        device=target_device,
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
    """Adjust UVs to point into packed texture regions.

    Args:
        verts_uvs: UV coordinates to be remapped.
        faces_uvs: Face-to-UV indices.
        map_offsets: Tensor of shape `(K, 3)` with `(y_offset, height, width)` per
            packed region.
        atlas_height: Height of the packed texture.
        atlas_width: Width of the packed texture.
        materials_idx: Per-face material or block index `(F,)`.

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
        assert isinstance(atlas_height, int), (
            "Expected `atlas_height` to be an `int`. " f"{type(atlas_height)=}"
        )
        assert isinstance(atlas_width, int), (
            "Expected `atlas_width` to be an `int`. " f"{type(atlas_width)=}"
        )
        assert isinstance(materials_idx, torch.Tensor), (
            "Expected `materials_idx` to be a tensor. " f"{type(materials_idx)=}"
        )
        assert map_offsets.shape[0] > 0, (
            "Expected `map_offsets` to contain at least one packed region. "
            f"{map_offsets.shape=}"
        )
        assert materials_idx.shape[0] == faces_uvs.shape[0], (
            "Expected one material id per face. "
            f"{materials_idx.shape=} {faces_uvs.shape=}"
        )

    _validate_inputs()

    flat_uv_indices = faces_uvs.reshape(-1)
    flat_uv_coords = verts_uvs.index_select(dim=0, index=flat_uv_indices).clone()
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
