from pathlib import Path
from typing import Union

import torch

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def save_ply_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None:
    """Write a Mesh to PLY, dispatched to the texture-representation-specific writer.

    PLY carries geometry plus optional per-vertex color; a UV-atlas texture has
    no PLY representation and is rejected.

    Args:
        mesh: `Mesh` instance to save.
        output_path: Output PLY filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )

    _validate_inputs()

    ply_path = _resolve_output_ply_path(output_path=output_path)
    ply_path.parent.mkdir(parents=True, exist_ok=True)

    if mesh.texture is None:
        _save_geometry_only_ply(mesh=mesh, ply_path=ply_path)
        return
    if isinstance(mesh.texture, MeshTextureVertexColor):
        _save_vertex_color_ply(mesh=mesh, ply_path=ply_path)
        return
    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        raise ValueError(
            "A UV-atlas texture cannot be written to PLY; save to OBJ or GLB "
            f"instead. {type(mesh.texture)=}"
        )
    assert 0, (
        "should not reach here: a Mesh texture is None, MeshTextureVertexColor, "
        f"or MeshTextureUVTextureMap. {type(mesh.texture)=}"
    )


def _save_geometry_only_ply(mesh: Mesh, ply_path: Path) -> None:
    """Write a geometry-only PLY.

    Args:
        mesh: Geometry-only `Mesh`.
        ply_path: Concrete PLY output filepath.

    Returns:
        None.
    """

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    with ply_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {verts_np.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {faces_np.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex_row in verts_np:
            handle.write(
                "{:.6f} {:.6f} {:.6f}\n".format(
                    float(vertex_row[0]),
                    float(vertex_row[1]),
                    float(vertex_row[2]),
                )
            )
        for face_row in faces_np:
            handle.write(
                "3 {} {} {}\n".format(
                    int(face_row[0]),
                    int(face_row[1]),
                    int(face_row[2]),
                )
            )


def _save_vertex_color_ply(mesh: Mesh, ply_path: Path) -> None:
    """Write a vertex-colored PLY.

    Args:
        mesh: `Mesh` carrying a `MeshTextureVertexColor`.
        ply_path: Concrete PLY output filepath.

    Returns:
        None.
    """

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    colors_np = _normalize_vertex_color_for_ply(
        vertex_color=mesh.texture.vertex_color
    ).numpy()
    with ply_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {verts_np.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write(f"element face {faces_np.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex_row, color_row in zip(verts_np, colors_np, strict=True):
            handle.write(
                "{:.6f} {:.6f} {:.6f} {} {} {}\n".format(
                    float(vertex_row[0]),
                    float(vertex_row[1]),
                    float(vertex_row[2]),
                    int(color_row[0]),
                    int(color_row[1]),
                    int(color_row[2]),
                )
            )
        for face_row in faces_np:
            handle.write(
                "3 {} {} {}\n".format(
                    int(face_row[0]),
                    int(face_row[1]),
                    int(face_row[2]),
                )
            )


def _resolve_output_ply_path(output_path: Union[str, Path]) -> Path:
    """Resolve an output path to a concrete `.ply` file path.

    Args:
        output_path: Output PLY filepath or output directory path.

    Returns:
        Concrete PLY output filepath (a `.ply` path, or `<dir>/mesh.ply`).
    """

    def _validate_inputs() -> None:
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    candidate_path = Path(output_path)
    if candidate_path.suffix.lower() == ".ply":
        return candidate_path
    assert candidate_path.suffix == "", (
        "Expected PLY mesh saving to target a `.ply` file or a directory-like "
        "path without a suffix. "
        f"{candidate_path=}"
    )
    return candidate_path / "mesh.ply"


def _normalize_vertex_color_for_ply(vertex_color: torch.Tensor) -> torch.Tensor:
    """Convert one vertex-color tensor to uint8 RGB `[0, 255]` for PLY export.

    Args:
        vertex_color: Vertex-color tensor in uint8 `[0, 255]` or float32
            `[0, 1]` form.

    Returns:
        Uint8 vertex-color tensor in `[0, 255]`.
    """

    def _validate_inputs() -> None:
        assert isinstance(vertex_color, torch.Tensor), (
            "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(vertex_color)=}"
        )
        assert vertex_color.dtype in (torch.uint8, torch.float32), (
            "Expected `vertex_color` to be uint8 or float32. " f"{vertex_color.dtype=}"
        )

    _validate_inputs()

    if vertex_color.dtype == torch.uint8:
        return vertex_color.contiguous()
    return vertex_color.mul(255.0).round().to(dtype=torch.uint8).contiguous()
