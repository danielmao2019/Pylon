"""Mesh loading utilities built around PyTorch3D."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


def load_meshes(
    mesh_root: Union[str, Path],
    device: Union[str, torch.device] = 'cuda',
    dtype: torch.dtype = torch.float32,
    load_textures: bool = True,
) -> Meshes:
    """Load all OBJ blocks within a directory and merge them into a single mesh."""
    assert load_textures, 'Texture baking requires textures to be loaded'
    root_path = Path(mesh_root)
    obj_paths = _discover_obj_paths(root_path)
    block_meshes = [
        _load_mesh_block(path, device=device, dtype=dtype) for path in obj_paths
    ]
    return _merge_blocks(block_meshes)


def _discover_obj_paths(root: Path) -> List[Path]:
    assert root.is_dir(), f"Mesh root must be a directory: {root}"
    top_level_objs = sorted(root.glob('*.obj'))
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    subdir_objs: List[Path] = [
        obj for subdir in subdirs for obj in sorted(subdir.glob('*.obj'))
    ]

    assert not (
        top_level_objs and subdir_objs
    ), f"Mesh root {root} should contain OBJ files either at the top level or within immediate subdirectories, not both"

    candidates: List[Path] = []
    if top_level_objs:
        candidates.extend(top_level_objs)
    elif subdir_objs:
        candidates.extend(subdir_objs)

    if not candidates:
        raise FileNotFoundError(f"No OBJ meshes discovered under directory: {root}")
    return candidates


def _load_mesh_block(
    obj_path: Path,
    device: Union[str, torch.device],
    dtype: torch.dtype,
) -> Meshes:
    assert obj_path.is_file(), f"Mesh file not found: {obj_path}"
    assert obj_path.suffix.lower() == '.obj'

    device = torch.device(device)

    material_names = _material_names_in_order(obj_path)
    assert material_names, 'Expected at least one material reference in OBJ'

    verts, faces, aux = load_obj(obj_path, load_textures=True)

    assert verts.numel() > 0, f"Loaded mesh has no geometry: {obj_path}"

    faces_idx = faces.verts_idx.to(dtype=torch.long)
    textures_idx = faces.textures_idx
    materials_idx = faces.materials_idx
    assert textures_idx is not None, 'Expected texture coordinates per face'
    assert materials_idx is not None, 'Expected material indices per face'

    textures_idx = textures_idx.to(dtype=torch.long)
    materials_idx = materials_idx.to(dtype=torch.long)
    assert faces_idx.shape[1] == 3 and textures_idx.shape[1] == 3
    assert torch.all(materials_idx >= 0)

    verts_uvs = aux.verts_uvs
    texture_images = aux.texture_images
    assert verts_uvs is not None
    assert texture_images

    max_material_index = int(materials_idx.max().item())
    assert max_material_index < len(material_names)

    material_textures: Dict[int, torch.Tensor] = {}
    for index, name in enumerate(material_names):
        if index > max_material_index:
            break
        assert name in texture_images, f"Texture for material '{name}' is missing"
        tensor = texture_images[name]
        assert isinstance(tensor, torch.Tensor)
        material_textures[index] = tensor.to(dtype=torch.float32)[..., :3]

    flat_vertex_indices = faces_idx.reshape(-1)
    flat_uv_indices = textures_idx.reshape(-1)
    flat_material_indices = materials_idx.repeat_interleave(3)

    expanded_vertices = verts.index_select(0, flat_vertex_indices).to(dtype=dtype)
    expanded_uvs = verts_uvs.index_select(0, flat_uv_indices).to(dtype=torch.float32)

    vertex_colors = torch.empty((expanded_vertices.shape[0], 3), dtype=torch.float32)
    assigned = torch.zeros(vertex_colors.shape[0], dtype=torch.bool)
    for material_index, texture in material_textures.items():
        mask = flat_material_indices == material_index
        if mask.any():
            sampled = _bilinear_sample(texture, expanded_uvs[mask])
            vertex_colors[mask] = sampled
            assigned |= mask

    assert assigned.all(), 'Every vertex should map to exactly one material texture'

    new_face_indices = torch.arange(expanded_vertices.shape[0], dtype=torch.long).view(
        -1, 3
    )

    verts_device = expanded_vertices.to(device=device)
    faces_device = new_face_indices.to(device=device)
    colors_device = vertex_colors.to(device=device)

    textures = TexturesVertex(verts_features=[colors_device])
    return Meshes(verts=[verts_device], faces=[faces_device], textures=textures)


def _material_names_in_order(path: Path) -> List[str]:
    names: List[str] = []
    with path.open('r', encoding='utf-8', errors='ignore') as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith('usemtl '):
                continue
            name = stripped.split(maxsplit=1)[1]
            if name and name not in names:
                names.append(name)
    return names


def _bilinear_sample(texture_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    assert texture_map.dim() == 3 and texture_map.shape[2] >= 3
    height, width = texture_map.shape[0], texture_map.shape[1]

    u = uv[:, 0].clamp(0.0, 1.0)
    v = uv[:, 1].clamp(0.0, 1.0)

    x = u * (width - 1)
    y = (1.0 - v) * (height - 1)

    x0 = torch.floor(x)
    y0 = torch.floor(y)
    x1 = torch.clamp(x0 + 1, max=width - 1)
    y1 = torch.clamp(y0 + 1, max=height - 1)

    x0l = x0.long()
    y0l = y0.long()
    x1l = x1.long()
    y1l = y1.long()

    wx = (x - x0).unsqueeze(1)
    wy = (y - y0).unsqueeze(1)

    c00 = texture_map[y0l, x0l]
    c01 = texture_map[y0l, x1l]
    c10 = texture_map[y1l, x0l]
    c11 = texture_map[y1l, x1l]

    top = c00 * (1.0 - wx) + c01 * wx
    bottom = c10 * (1.0 - wx) + c11 * wx
    return top * (1.0 - wy) + bottom * wy


def _merge_blocks(block_meshes: Sequence[Meshes]) -> Meshes:
    assert block_meshes, 'Expected at least one block mesh to merge'
    if len(block_meshes) == 1:
        return block_meshes[0]

    verts_list: List[torch.Tensor] = []
    faces_list: List[torch.Tensor] = []
    colors_list: List[torch.Tensor] = []
    vertex_offset = 0
    for mesh in block_meshes:
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0] + vertex_offset
        textures = mesh.textures.verts_features_list()[0]
        verts_list.append(verts)
        faces_list.append(faces)
        colors_list.append(textures)
        vertex_offset += verts.shape[0]

    merged_verts = torch.cat(verts_list, dim=0)
    merged_faces = torch.cat(faces_list, dim=0)
    merged_colors = torch.cat(colors_list, dim=0)

    textures = TexturesVertex(verts_features=[merged_colors])
    return Meshes(verts=[merged_verts], faces=[merged_faces], textures=textures)
