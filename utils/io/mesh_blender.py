"""Blender-native mesh loading utilities mirroring PyTorch3D helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import bpy


def _discover_obj_paths_blender(root: Path) -> List[Path]:
    assert root.is_dir(), f"Mesh root must be a directory: {root}"
    top_level = sorted(root.glob('*.obj'))
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    nested = [obj for sub in subdirs for obj in sorted(sub.glob('*.obj'))]
    assert not (
        top_level and nested
    ), (
        f"Mesh root {root} should contain OBJ files either at the top level "
        "or within immediate subdirectories, not both"
    )
    candidates: List[Path] = top_level or nested
    if not candidates:
        raise FileNotFoundError(f"No OBJ meshes discovered under directory: {root}")
    return candidates


def _ensure_collection_blender(
    collection_name: str,
    clear: bool,
    link_to_scene: bool,
):
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        if clear:
            for obj in list(collection.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        collection = bpy.data.collections.new(collection_name)

    if link_to_scene:
        scene = bpy.context.scene
        if collection.name not in scene.collection.children:
            scene.collection.children.link(collection)
    return collection


def _import_single_obj_blender(
    obj_path: Path,
    collection,
) -> Sequence[str]:
    before = set(bpy.data.objects.keys())
    bpy.ops.import_scene.obj(
        filepath=str(obj_path),
        axis_forward='-Z',
        axis_up='Y',
        split_mode='OFF',
    )
    after = set(bpy.data.objects.keys())
    new_names = [name for name in after - before]

    linked: List[str] = []
    for name in new_names:
        obj = bpy.data.objects[name]
        if obj.type != 'MESH':
            continue
        if collection not in obj.users_collection:
            collection.objects.link(obj)
        obj.hide_render = False
        obj.hide_viewport = False
        obj.pass_index = 0
        linked.append(obj.name)
    return linked


_DEFAULT_COLLECTION = 'mesh_collection_blender'


def load_meshes_blender(
    mesh_root: Union[str, Path],
) -> Dict[str, Union[str, List[str]]]:
    """Import OBJ meshes into Blender collections analogous to PyTorch3D loader."""
    root_path = Path(mesh_root)
    obj_paths = _discover_obj_paths_blender(root_path)
    collection = _ensure_collection_blender(
        collection_name=_DEFAULT_COLLECTION,
        clear=True,
        link_to_scene=True,
    )

    imported: List[str] = []
    for path in obj_paths:
        new_objects = _import_single_obj_blender(path, collection)
        imported.extend(new_objects)

    if not imported:
        raise RuntimeError(f"No mesh objects were imported from {mesh_root}")

    return {
        'collection_name': collection.name,
        'object_names': imported,
    }
