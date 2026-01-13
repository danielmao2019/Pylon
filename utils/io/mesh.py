"""Mesh loading utilities built around PyTorch3D."""

from pathlib import Path
from typing import List, Sequence, Union

import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes


def load_meshes(
    mesh_root: Union[str, Path],
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    load_textures: bool = True,
) -> Meshes:
    """Load one or more OBJ meshes under a root directory and return a merged PyTorch3D mesh.

    Args:
        mesh_root: Directory containing one or more OBJ files (either at top level or one level below).
        device: Target device for tensors.
        dtype: Target dtype for vertex positions.
        load_textures: Must be True; textures are required for rendering.

    Returns:
        A single `Meshes` instance containing all discovered OBJ blocks. Loading and
        merging are performed on CPU; the merged mesh is moved to `device` once at the
        end.
    """
    assert load_textures, 'Texture baking requires textures to be loaded'
    mesh_root_path = Path(mesh_root)
    obj_files = _discover_obj_paths(mesh_root_path)

    block_meshes = [_load_mesh_block(path, dtype=dtype) for path in obj_files]
    merged_cpu = _merge_blocks(block_meshes)
    return merged_cpu.to(device=torch.device(device))


def _discover_obj_paths(root: Path) -> List[Path]:
    """Enumerate OBJ files in a mesh root, allowing either top-level OBJs or immediate subdirectory OBJs."""
    assert root.is_dir(), f"Mesh root must be a directory: {root}"

    # --- Gather candidate OBJ paths at the root and one level below
    top_level_obj_paths = sorted(root.glob('*.obj'))
    subdirectories = sorted(path for path in root.iterdir() if path.is_dir())
    nested_obj_paths: List[Path] = [
        obj for subdir in subdirectories for obj in sorted(subdir.glob('*.obj'))
    ]

    assert not (
        top_level_obj_paths and nested_obj_paths
    ), f"Mesh root {root} should contain OBJ files either at the top level or within immediate subdirectories, not both"

    candidates: List[Path] = []
    if top_level_obj_paths:
        candidates.extend(top_level_obj_paths)
    elif nested_obj_paths:
        candidates.extend(nested_obj_paths)

    if not candidates:
        raise FileNotFoundError(f"No OBJ meshes discovered under directory: {root}")
    return candidates


def _load_mesh_block(
    obj_path: Path,
    dtype: torch.dtype,
) -> Meshes:
    """Load a single OBJ with textures and return a PyTorch3D mesh with packed UV textures.

    Args:
        obj_path: Path to the OBJ file.
        dtype: Target dtype for vertex positions.

    Returns:
        A `Meshes` instance with `TexturesUV` referencing a packed texture map.
    """
    assert obj_path.is_file(), f"Mesh file not found: {obj_path}"
    assert obj_path.suffix.lower() == '.obj'

    # --- Load OBJ content on CPU
    verts, faces, aux = load_obj(
        obj_path,
        load_textures=True,
        device=torch.device('cpu'),
    )

    # --- Validate required mesh components
    assert verts.numel() > 0, f"Loaded mesh has no geometry: {obj_path}"
    assert faces.textures_idx is not None, 'Expected texture coordinates per face'
    assert faces.materials_idx is not None, 'Expected material indices per face'
    assert aux.verts_uvs is not None, 'Expected UV coordinates for textured mesh'
    assert aux.texture_images, 'Expected texture images for materials'
    assert faces.verts_idx.shape[1] == 3
    assert faces.textures_idx.shape[1] == 3

    faces_idx = faces.verts_idx.to(dtype=torch.long)
    textures_idx = faces.textures_idx.to(dtype=torch.long)
    verts_uvs = aux.verts_uvs.to(dtype=torch.float32)
    materials_idx = faces.materials_idx.to(dtype=torch.long)
    assert torch.all(materials_idx >= 0)
    assert aux.texture_images, 'Expected at least one texture image'

    # --- Pack material textures into a single atlas and remap UVs
    packed_map, remapped_uvs, remapped_faces_uvs = _pack_textures_uvs(
        texture_images=aux.texture_images,
        verts_uvs=verts_uvs,
        faces_uvs=textures_idx,
        materials_idx=materials_idx,
    )

    textures_uv = TexturesUV(
        maps=[packed_map],
        faces_uvs=[remapped_faces_uvs],
        verts_uvs=[remapped_uvs],
    )
    verts_cpu = verts.to(dtype=dtype)
    return Meshes(verts=[verts_cpu], faces=[faces_idx], textures=textures_uv)


def _pack_textures_uvs(
    texture_images: dict[str, torch.Tensor],
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    materials_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack all material textures from one OBJ into a single map and remap UVs.

    Textures are stacked top-to-bottom in the atlas; offsets are measured from the
    bottom so v=0 stays at the bottom of each packed region.

    Args:
        texture_images: Mapping of material name to texture tensor (H, W, C).
        verts_uvs: UV coordinates for the OBJ.
        faces_uvs: Face-to-UV indices (F, 3).
        materials_idx: Face-to-material indices (F,).

    Returns:
        packed_map: Combined texture map containing all materials.
        remapped_uvs: UV coordinates adjusted to the combined map.
        remapped_faces_uvs: Dense face UV indices aligned with remapped_uvs.
    """
    material_names = list(texture_images.keys())
    assert material_names, 'Expected at least one material name'

    # --- Collect RGB textures on the same device as UVs
    target_device = verts_uvs.device
    material_textures: List[torch.Tensor] = []
    for name in material_names:
        texture = texture_images[name]
        assert texture.dim() == 3 and texture.shape[2] >= 3
        material_textures.append(texture[..., :3].to(device=target_device))

    # --- Determine packed atlas dimensions
    atlas_height = sum(int(tex.shape[0]) for tex in material_textures)
    atlas_width = max(int(tex.shape[1]) for tex in material_textures)
    assert atlas_height > 0 and atlas_width > 0

    atlas_device = material_textures[0].device
    atlas_dtype = material_textures[0].dtype
    packed_map = torch.zeros(
        (atlas_height, atlas_width, 3), device=atlas_device, dtype=atlas_dtype
    )

    # --- Place textures top-to-bottom into the atlas and record offsets
    y_offset = 0
    offset_entries: List[torch.Tensor] = []
    for texture in material_textures:
        height = int(texture.shape[0])
        width = int(texture.shape[1])

        packed_map[y_offset : y_offset + height, 0:width, :] = texture
        bottom_offset = float(atlas_height - y_offset - height)
        offset_entries.append(
            torch.tensor(
                [bottom_offset, float(height), float(width)],
                device=atlas_device,
                dtype=torch.float32,
            )
        )
        y_offset += height

    offset_table = torch.stack(offset_entries, dim=0)

    remapped_uvs = _remap_uvs(
        verts_uvs=verts_uvs,
        faces_uvs=faces_uvs,
        map_offsets=offset_table,
        atlas_height=atlas_height,
        atlas_width=atlas_width,
        materials_idx=materials_idx,
    )

    flat_count = faces_uvs.numel()
    new_faces_uvs = torch.arange(
        flat_count, device=atlas_device, dtype=torch.long
    ).view(-1, 3)

    return packed_map, remapped_uvs, new_faces_uvs


def _merge_blocks(block_meshes: Sequence[Meshes]) -> Meshes:
    """Merge multiple block meshes into one mesh while preserving UV textures.

    Textures are packed into one atlas by stacking them top-to-bottom; UV offsets are
    recorded from the bottom so v=0 stays at the bottom of each region.

    Args:
        block_meshes: Sequence of per-block `Meshes` objects with `TexturesUV`.

    Returns:
        A single `Meshes` with concatenated geometry and packed texture map.
    """
    assert block_meshes, 'Expected at least one block mesh to merge'
    if len(block_meshes) == 1:
        return block_meshes[0]

    verts_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    verts_uv_list: List[torch.Tensor] = []
    faces_uv_list: List[torch.Tensor] = []
    texture_maps: List[torch.Tensor] = []
    face_materials_list: List[torch.Tensor] = []
    vertex_offset = 0
    uv_offset = 0
    atlas_width = 0

    # --- Collect geometry and textures from each block mesh
    for block_index, mesh in enumerate(block_meshes):
        assert len(mesh.verts_list()) == 1
        assert len(mesh.faces_list()) == 1
        assert isinstance(mesh.textures, TexturesUV)
        assert len(mesh.textures.maps_list()) == 1
        assert len(mesh.textures.verts_uvs_list()) == 1
        assert len(mesh.textures.faces_uvs_list()) == 1

        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0] + vertex_offset

        map_tensor = mesh.textures.maps_list()[0]
        assert map_tensor.dim() == 3 and map_tensor.shape[2] == 3

        verts_uvs = mesh.textures.verts_uvs_list()[0]
        faces_uvs = mesh.textures.faces_uvs_list()[0] + uv_offset

        verts_list.append(verts)
        faces_list.append(faces)
        verts_uv_list.append(verts_uvs)
        faces_uv_list.append(faces_uvs)
        face_materials_list.append(
            torch.full(
                (faces.shape[0],),
                block_index,
                device=faces.device,
                dtype=torch.long,
            )
        )

        height = int(map_tensor.shape[0])
        width = int(map_tensor.shape[1])
        texture_maps.append(map_tensor)
        atlas_width = max(atlas_width, width)

        vertex_offset += verts.shape[0]
        uv_offset += verts_uvs.shape[0]

    # --- Concatenate block geometry and UV data
    merged_verts = torch.cat(verts_list, dim=0)
    merged_faces = torch.cat(faces_list, dim=0)
    merged_verts_uvs = torch.cat(verts_uv_list, dim=0)
    merged_faces_uvs = torch.cat(faces_uv_list, dim=0)
    merged_face_materials = torch.cat(face_materials_list, dim=0)

    # --- Pack all texture maps into a single atlas
    atlas_height = sum(int(tex.shape[0]) for tex in texture_maps)
    assert atlas_height > 0 and atlas_width > 0
    target_device = merged_verts_uvs.device
    packed_map = torch.zeros(
        (atlas_height, atlas_width, 3),
        dtype=texture_maps[0].dtype,
        device=target_device,
    )

    y_offset = 0
    remap_entries: List[torch.Tensor] = []
    for map_tensor in texture_maps:
        height = int(map_tensor.shape[0])
        width = int(map_tensor.shape[1])
        packed_map[y_offset : y_offset + height, 0:width, :] = map_tensor.to(
            device=target_device
        )
        bottom_offset = float(atlas_height - y_offset - height)
        remap_entries.append(
            torch.tensor(
                [bottom_offset, float(height), float(width)],
                device=target_device,
                dtype=torch.float32,
            )
        )
        y_offset += height

    remap_stack = torch.stack(remap_entries, dim=0)
    assert remap_stack.shape[0] == len(texture_maps)

    merged_verts_uvs = _remap_uvs(
        verts_uvs=merged_verts_uvs,
        faces_uvs=merged_faces_uvs,
        map_offsets=remap_stack,
        atlas_height=atlas_height,
        atlas_width=atlas_width,
        materials_idx=merged_face_materials,
    )

    merged_textures = TexturesUV(
        maps=[packed_map],
        faces_uvs=[merged_faces_uvs],
        verts_uvs=[merged_verts_uvs],
    )
    return Meshes(verts=[merged_verts], faces=[merged_faces], textures=merged_textures)


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
        map_offsets: Tensor of shape (K, 3) with (y_offset, height, width) per region.
        atlas_height: Height of the packed texture.
        atlas_width: Width of the packed texture.
        materials_idx: Per-face material/block index (F,).

    Returns:
        Remapped UV coordinates aligned to the packed texture.
    """
    assert map_offsets.shape[0] > 0
    assert materials_idx.shape[0] == faces_uvs.shape[0]

    flat_indices = faces_uvs.reshape(-1)
    flat_uvs = verts_uvs.index_select(0, flat_indices)
    face_materials_expanded = materials_idx.repeat_interleave(3)

    remapped_uvs = flat_uvs.clone()
    material_ids = torch.unique(face_materials_expanded)
    for material_id in material_ids:
        material_index = int(material_id.item())
        assert material_index < map_offsets.shape[0]
        mask = face_materials_expanded == material_id
        offset_y, height, width = map_offsets[material_index]

        remapped_material = remapped_uvs[mask]
        remapped_material[:, 0] = (remapped_material[:, 0] * width) / float(atlas_width)
        remapped_material[:, 1] = (remapped_material[:, 1] * height + offset_y) / float(
            atlas_height
        )
        remapped_uvs[mask] = remapped_material

    return remapped_uvs
