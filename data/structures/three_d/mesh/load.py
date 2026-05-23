from pathlib import Path
from typing import Dict, List, Union

import torch
from pytorch3d.io import load_obj

from data.structures.three_d.mesh.merge import merge_meshes, pack_texture_images
from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def load_mesh(path: Union[str, Path]) -> Mesh:
    """Load one OBJ file, or every OBJ under a mesh-root directory, as one Mesh.

    Args:
        path: Mesh file path or supported mesh-root directory path.

    Returns:
        One merged `Mesh`.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )

    _validate_inputs()

    obj_paths = _resolve_input_paths(path=path)
    mesh_blocks = [
        _load_mesh_block_from_obj_path(obj_path=obj_path) for obj_path in obj_paths
    ]
    return merge_meshes(mesh_blocks=mesh_blocks)


def _load_mesh_block_from_obj_path(obj_path: Path) -> Mesh:
    """Load one OBJ as a single mesh block.

    Dispatches to the texture-representation-specific loader.

    Args:
        obj_path: Concrete OBJ filepath.

    Returns:
        One `Mesh` block.
    """

    def _validate_inputs() -> None:
        assert isinstance(obj_path, Path), (
            "Expected `obj_path` to be a `Path`. " f"{type(obj_path)=}"
        )

    _validate_inputs()

    obj_features = _inspect_obj_file(obj_path=obj_path)
    if obj_features["has_uv_coords"] and obj_features["has_uv_faces"]:
        return _load_mesh_uv_texture_map(path=obj_path)
    if obj_features["has_vertex_colors"]:
        return _load_mesh_vertex_color(path=obj_path)
    return _load_mesh_geometry_only(path=obj_path)


def _load_mesh_geometry_only(path: Union[str, Path]) -> Mesh:
    """Load one geometry-only OBJ mesh.

    Args:
        path: Geometry-only OBJ filepath or directory path that resolves to one.

    Returns:
        One geometry-only `Mesh` (texture `None`).
    """

    obj_path = _resolve_input_path(path=path)

    vertex_rows: List[List[float]] = []
    face_rows: List[List[int]] = []
    with obj_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.startswith("v "):
                vertex_parts = line.split()
                assert len(vertex_parts) >= 4, (
                    "Expected geometry-only OBJ verts to include xyz values. "
                    f"{obj_path=} {line=}"
                )
                vertex_rows.append(
                    [
                        float(vertex_parts[1]),
                        float(vertex_parts[2]),
                        float(vertex_parts[3]),
                    ]
                )
                continue
            if line.startswith("f "):
                face_parts = line.split()[1:]
                assert len(face_parts) == 3, (
                    "Expected geometry-only OBJ loading to receive triangular "
                    "faces. "
                    f"{obj_path=} {line=}"
                )
                face_rows.append(
                    [
                        int(face_parts[0].split("/")[0]) - 1,
                        int(face_parts[1].split("/")[0]) - 1,
                        int(face_parts[2].split("/")[0]) - 1,
                    ]
                )

    assert vertex_rows, (
        "Expected geometry-only OBJ loading to find at least one vertex. "
        f"{obj_path=}"
    )
    assert face_rows, (
        "Expected geometry-only OBJ loading to find at least one face. " f"{obj_path=}"
    )

    return Mesh(
        verts=torch.tensor(vertex_rows, dtype=torch.float32).contiguous(),
        faces=torch.tensor(face_rows, dtype=torch.int64).contiguous(),
        texture=None,
    )


def _load_mesh_vertex_color(path: Union[str, Path]) -> Mesh:
    """Load one vertex-colored OBJ mesh.

    Args:
        path: Vertex-colored OBJ filepath or directory path that resolves to
            one.

    Returns:
        One `Mesh` carrying a `MeshTextureVertexColor`.
    """

    obj_path = _resolve_input_path(path=path)

    vertex_rows: List[List[float]] = []
    color_rows: List[List[float]] = []
    face_rows: List[List[int]] = []
    with obj_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.startswith("v "):
                vertex_parts = line.split()
                assert len(vertex_parts) >= 7, (
                    "Expected vertex-colored OBJ verts to include RGB values. "
                    f"{obj_path=} {line=}"
                )
                vertex_rows.append(
                    [
                        float(vertex_parts[1]),
                        float(vertex_parts[2]),
                        float(vertex_parts[3]),
                    ]
                )
                color_rows.append(
                    [
                        float(vertex_parts[4]),
                        float(vertex_parts[5]),
                        float(vertex_parts[6]),
                    ]
                )
                continue
            if line.startswith("f "):
                face_parts = line.split()[1:]
                assert len(face_parts) == 3, (
                    "Expected vertex-colored OBJ loading to receive triangular "
                    "faces. "
                    f"{obj_path=} {line=}"
                )
                face_rows.append(
                    [
                        int(face_parts[0].split("/")[0]) - 1,
                        int(face_parts[1].split("/")[0]) - 1,
                        int(face_parts[2].split("/")[0]) - 1,
                    ]
                )

    assert vertex_rows, (
        "Expected vertex-colored OBJ loading to find at least one vertex. "
        f"{obj_path=}"
    )
    assert face_rows, (
        "Expected vertex-colored OBJ loading to find at least one face. " f"{obj_path=}"
    )
    assert len(color_rows) == len(vertex_rows), (
        "Expected vertex-colored OBJ loading to find one RGB value per vertex. "
        f"{obj_path=} {len(color_rows)=} {len(vertex_rows)=}"
    )

    vertex_color = torch.tensor(color_rows, dtype=torch.float32)
    if float(vertex_color.max().item()) > 1.0:
        vertex_color = vertex_color / 255.0
    return Mesh(
        verts=torch.tensor(vertex_rows, dtype=torch.float32).contiguous(),
        faces=torch.tensor(face_rows, dtype=torch.int64).contiguous(),
        texture=MeshTextureVertexColor(vertex_color=vertex_color.contiguous()),
    )


def _load_mesh_uv_texture_map(path: Union[str, Path]) -> Mesh:
    """Load one UV-textured OBJ mesh.

    Loads via PyTorch3D, which already yields the decoupled geometry/UV
    domains, so no welding is needed. The UV-origin convention is `"obj"`.

    Args:
        path: UV-textured OBJ filepath or directory path that resolves to one.

    Returns:
        One `Mesh` carrying a `MeshTextureUVTextureMap` on the geometry domain.
    """

    obj_path = _resolve_input_path(path=path)

    verts, faces, aux = load_obj(
        obj_path,
        load_textures=True,
        device=torch.device("cpu"),
    )
    assert aux.verts_uvs is not None, (
        "Expected UV-textured OBJ loading to provide `verts_uvs`. " f"{obj_path=}"
    )
    assert faces.textures_idx is not None, (
        "Expected UV-textured OBJ loading to provide `textures_idx`. " f"{obj_path=}"
    )
    assert aux.texture_images, (
        "Expected UV-textured OBJ loading to provide at least one texture "
        "image via standard OBJ/MTL relative-path resolution. "
        f"{obj_path=}"
    )

    verts_uvs = aux.verts_uvs.to(dtype=torch.float32).contiguous()
    faces_uvs = faces.textures_idx.to(dtype=torch.int64).contiguous()
    if len(aux.texture_images) == 1:
        only_texture_name = list(aux.texture_images.keys())[0]
        uv_texture_map = aux.texture_images[only_texture_name][..., :3].to(
            dtype=torch.float32
        )
    else:
        assert faces.materials_idx is not None, (
            "Expected multi-material OBJ loading to provide `materials_idx`. "
            f"{obj_path=}"
        )
        uv_texture_map, verts_uvs, faces_uvs = pack_texture_images(
            texture_images=aux.texture_images,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            materials_idx=faces.materials_idx.to(dtype=torch.int64).contiguous(),
        )

    return Mesh(
        verts=verts.to(dtype=torch.float32).contiguous(),
        faces=faces.verts_idx.to(dtype=torch.int64).contiguous(),
        texture=MeshTextureUVTextureMap(
            uv_texture_map=uv_texture_map.detach().cpu().contiguous(),
            verts_uvs=verts_uvs.detach().cpu().contiguous(),
            faces_uvs=faces_uvs.detach().cpu().contiguous(),
            convention="obj",
        ),
    )


def _resolve_input_path(path: Union[str, Path]) -> Path:
    """Resolve one user mesh path to exactly one OBJ file.

    Args:
        path: Mesh file path or supported mesh-root directory path.

    Returns:
        Concrete OBJ filepath.
    """

    candidate_obj_paths = _resolve_input_paths(path=path)
    assert len(candidate_obj_paths) == 1, (
        "Expected the mesh loading path to resolve to exactly one OBJ file. "
        f"{path=} {candidate_obj_paths=}"
    )
    return candidate_obj_paths[0]


def _resolve_input_paths(path: Union[str, Path]) -> List[Path]:
    """Resolve one user mesh path to one or more OBJ file paths.

    Args:
        path: Mesh file path or supported mesh-root directory path.

    Returns:
        Concrete OBJ file paths (one OBJ file, or every OBJ at the top level or
        one level below a directory).
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )
        assert Path(path).exists(), (
            "Expected `path` to exist before mesh loading. " f"{Path(path)=}"
        )

    _validate_inputs()

    candidate_path = Path(path)
    if candidate_path.is_file():
        assert candidate_path.suffix.lower() == ".obj", (
            "Expected mesh file loading to receive an `.obj` path. "
            f"{candidate_path=}"
        )
        return [candidate_path]

    assert candidate_path.is_dir(), (
        "Expected the mesh loading path to be either a file or a directory. "
        f"{candidate_path=}"
    )

    top_level_obj_paths = sorted(candidate_path.glob("*.obj"))
    nested_obj_paths = sorted(candidate_path.glob("*/*.obj"))
    assert not (top_level_obj_paths and nested_obj_paths), (
        "Expected mesh directory loading to contain OBJ files either at the top "
        "level or one level below, not both. "
        f"{candidate_path=} {top_level_obj_paths=} {nested_obj_paths=}"
    )
    candidate_obj_paths = top_level_obj_paths + nested_obj_paths
    assert candidate_obj_paths, (
        "Expected mesh directory loading to find at least one OBJ file. "
        f"{candidate_path=}"
    )
    return candidate_obj_paths


def _inspect_obj_file(obj_path: Path) -> Dict[str, bool]:
    """Inspect one OBJ file to determine its texture representation.

    Args:
        obj_path: Concrete OBJ filepath.

    Returns:
        Dictionary with `has_vertex_colors` / `has_uv_coords` / `has_uv_faces` /
        `has_mtllib` flags.
    """

    def _validate_inputs() -> None:
        assert isinstance(obj_path, Path), (
            "Expected `obj_path` to be a `Path`. " f"{type(obj_path)=}"
        )
        assert obj_path.is_file(), (
            "Expected `obj_path` to exist as a file. " f"{obj_path=}"
        )

    _validate_inputs()

    has_vertex_colors = False
    has_uv_coords = False
    has_uv_faces = False
    has_mtllib = False
    with obj_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "" or line.startswith("#"):
                continue
            if line.startswith("mtllib "):
                has_mtllib = True
                continue
            if line.startswith("vt "):
                has_uv_coords = True
                continue
            if line.startswith("v "):
                if len(line.split()) >= 7:
                    has_vertex_colors = True
                continue
            if line.startswith("f "):
                face_parts = line.split()[1:]
                if any("/" in face_part for face_part in face_parts):
                    has_uv_faces = True

    return {
        "has_vertex_colors": has_vertex_colors,
        "has_uv_coords": has_uv_coords,
        "has_uv_faces": has_uv_faces,
        "has_mtllib": has_mtllib,
    }
