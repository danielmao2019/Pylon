from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import torch
from pytorch3d.io import load_obj

from data.structures.three_d.mesh.merge import merge_meshes, pack_texture_images

if TYPE_CHECKING:
    from data.structures.three_d.mesh.mesh import Mesh


def load_mesh(path: Union[str, Path]) -> Dict[str, Union[torch.Tensor, str, None]]:
    """Load one mesh from a supported file or directory path.

    Args:
        path: Mesh file path or supported mesh-root directory path.

    Returns:
        Normalized keyword arguments for `Mesh.__init__`.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )

    _validate_inputs()

    from data.structures.three_d.mesh.mesh import Mesh

    obj_paths = _resolve_input_paths(path=path)
    mesh_blocks = [
        Mesh(**_load_mesh_attributes_from_obj_path(obj_path=obj_path))
        for obj_path in obj_paths
    ]
    merged_mesh = merge_meshes(mesh_blocks=mesh_blocks)
    return _mesh_to_init_kwargs(mesh=merged_mesh)


def _load_mesh_attributes_from_obj_path(
    obj_path: Path,
) -> Dict[str, Union[torch.Tensor, str, None]]:
    """Load one OBJ file into mesh constructor kwargs.

    Args:
        obj_path: Concrete OBJ filepath.

    Returns:
        Normalized keyword arguments for `Mesh.__init__`.
    """

    obj_features = _inspect_obj_file(obj_path=obj_path)
    if obj_features["has_uv_coords"] and obj_features["has_uv_faces"]:
        return _load_mesh_uv_texture_map(path=obj_path)
    if obj_features["has_vertex_colors"]:
        return _load_mesh_vertex_color(path=obj_path)
    return _load_mesh_geometry_only(path=obj_path)


def _mesh_to_init_kwargs(mesh: "Mesh") -> Dict[str, Union[torch.Tensor, str, None]]:
    """Convert one repo mesh into `Mesh.__init__` keyword arguments.

    Args:
        mesh: Repo mesh instance.

    Returns:
        Normalized keyword arguments for `Mesh.__init__`.
    """

    return {
        "vertices": mesh.vertices.detach().cpu().contiguous(),
        "faces": mesh.faces.detach().cpu().contiguous(),
        "vertex_color": (
            None
            if mesh.vertex_color is None
            else mesh.vertex_color.detach().cpu().contiguous()
        ),
        "uv_texture_map": (
            None
            if mesh.uv_texture_map is None
            else mesh.uv_texture_map.detach().cpu().contiguous()
        ),
        "vertex_uv": (
            None
            if mesh.vertex_uv is None
            else mesh.vertex_uv.detach().cpu().contiguous()
        ),
        "face_uvs": (
            None if mesh.face_uvs is None else mesh.face_uvs.detach().cpu().contiguous()
        ),
        "convention": mesh.convention,
    }


def _load_mesh_geometry_only(
    path: Union[str, Path],
) -> Dict[str, Union[torch.Tensor, str, None]]:
    """Load one geometry-only OBJ mesh.

    Args:
        path: Geometry-only OBJ filepath or directory path that resolves to one.

    Returns:
        Normalized keyword arguments for one geometry-only mesh.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Path:
        return _resolve_input_path(path=path)

    obj_path = _normalize_inputs()

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
                    "Expected geometry-only OBJ vertices to include xyz values. "
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
                    "Expected geometry-only OBJ loading to receive triangular faces. "
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

    return {
        "vertices": torch.tensor(vertex_rows, dtype=torch.float32).contiguous(),
        "faces": torch.tensor(face_rows, dtype=torch.int64).contiguous(),
        "vertex_color": None,
        "uv_texture_map": None,
        "vertex_uv": None,
        "face_uvs": None,
        "convention": None,
    }


def _load_mesh_vertex_color(
    path: Union[str, Path],
) -> Dict[str, Union[torch.Tensor, str, None]]:
    """Load one vertex-colored OBJ mesh.

    Args:
        path: Vertex-colored OBJ filepath or directory path that resolves to one.

    Returns:
        Normalized keyword arguments for one vertex-colored mesh.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Path:
        return _resolve_input_path(path=path)

    obj_path = _normalize_inputs()

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
                    "Expected vertex-colored OBJ vertices to include RGB values. "
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
                    "Expected vertex-colored OBJ loading to receive triangular faces. "
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

    vertices = torch.tensor(vertex_rows, dtype=torch.float32)
    faces = torch.tensor(face_rows, dtype=torch.int64)
    vertex_color = torch.tensor(color_rows, dtype=torch.float32)
    if float(vertex_color.max().item()) > 1.0:
        vertex_color = vertex_color / 255.0
    return {
        "vertices": vertices.contiguous(),
        "faces": faces.contiguous(),
        "vertex_color": vertex_color.contiguous(),
        "uv_texture_map": None,
        "vertex_uv": None,
        "face_uvs": None,
        "convention": None,
    }


def _load_mesh_uv_texture_map(
    path: Union[str, Path],
) -> Dict[str, Union[torch.Tensor, str, None]]:
    """Load one UV-textured OBJ mesh.

    Args:
        path: UV-textured OBJ filepath or directory path that resolves to one.

    Returns:
        Normalized keyword arguments for one UV-textured mesh.
    """

    def _validate_inputs() -> None:
        assert isinstance(path, (str, Path)), (
            "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Path:
        return _resolve_input_path(path=path)

    obj_path = _normalize_inputs()

    vertices, faces, aux = load_obj(
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
        "Expected UV-textured OBJ loading to provide at least one texture image via "
        "standard OBJ/MTL relative-path resolution. "
        f"{obj_path=}"
    )
    mesh_vertices = vertices.to(dtype=torch.float32).contiguous()
    mesh_faces = faces.verts_idx.to(dtype=torch.int64).contiguous()
    vertex_uv = aux.verts_uvs.to(dtype=torch.float32).contiguous()
    face_uvs = faces.textures_idx.to(dtype=torch.int64).contiguous()

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
        uv_texture_map, vertex_uv, face_uvs = pack_texture_images(
            texture_images=aux.texture_images,
            verts_uvs=vertex_uv,
            faces_uvs=face_uvs,
            materials_idx=faces.materials_idx.to(dtype=torch.int64).contiguous(),
        )

    return {
        "vertices": mesh_vertices,
        "faces": mesh_faces,
        "vertex_color": None,
        "uv_texture_map": uv_texture_map.detach().cpu().contiguous(),
        "vertex_uv": vertex_uv.detach().cpu().contiguous(),
        "face_uvs": face_uvs.detach().cpu().contiguous(),
        "convention": "obj",
    }


def _resolve_input_path(path: Union[str, Path]) -> Path:
    """Resolve one user mesh path to a concrete OBJ file path.

    Args:
        path: Mesh file path or supported mesh-root directory path.

    Returns:
        Concrete OBJ filepath.
    """

    candidate_obj_paths = _resolve_input_paths(path=path)
    assert len(candidate_obj_paths) == 1, (
        "Expected mesh loading path to resolve to exactly one OBJ file. "
        f"{path=} {candidate_obj_paths=}"
    )
    return candidate_obj_paths[0]


def _resolve_input_paths(path: Union[str, Path]) -> List[Path]:
    """Resolve one user mesh path to one or more OBJ file paths.

    Args:
        path: Mesh file path or supported mesh-root directory path.

    Returns:
        Concrete OBJ file paths.
    """

    assert isinstance(path, (str, Path)), (
        "Expected `path` to be a `str` or `Path`. " f"{type(path)=}"
    )

    candidate_path = Path(path)
    assert candidate_path.exists(), (
        "Expected `path` to exist before mesh loading. " f"{candidate_path=}"
    )

    if candidate_path.is_file():
        assert candidate_path.suffix.lower() == ".obj", (
            "Expected mesh file loading to receive an `.obj` path. "
            f"{candidate_path=}"
        )
        return [candidate_path]

    assert candidate_path.is_dir(), (
        "Expected mesh loading path to be either a file or a directory. "
        f"{candidate_path=}"
    )

    top_level_obj_paths = sorted(candidate_path.glob("*.obj"))
    nested_obj_paths = sorted(candidate_path.glob("*/*.obj"))
    candidate_obj_paths = top_level_obj_paths + nested_obj_paths
    assert not (top_level_obj_paths and nested_obj_paths), (
        "Expected mesh directory loading to contain OBJ files either at the top "
        "level or one level below, not both. "
        f"{candidate_path=} {top_level_obj_paths=} {nested_obj_paths=}"
    )
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
        Dictionary describing the detected OBJ features.
    """

    assert isinstance(
        obj_path, Path
    ), f"Expected `obj_path` to be a `Path`. {type(obj_path)=}"
    assert obj_path.is_file(), f"Expected `obj_path` to exist as a file. {obj_path=}"

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
                vertex_parts = line.split()
                if len(vertex_parts) >= 7:
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
