from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import torch
from PIL import Image

from data.structures.three_d.mesh.conventions import transform_vertex_uv_convention
from data.structures.three_d.mesh.validate import validate_mesh_uv_convention

if TYPE_CHECKING:
    from data.structures.three_d.mesh.mesh import Mesh


def save_mesh(mesh: "Mesh", output_path: Union[str, Path]) -> None:
    """Save one mesh to disk.

    Args:
        mesh: `Mesh` instance to save.
        output_path: Output mesh filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

        assert isinstance(mesh, Mesh), (
            "Expected `mesh` saving input to be a `Mesh` instance. " f"{type(mesh)=}"
        )
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    if mesh.vertex_color is not None:
        _save_mesh_vertex_color(mesh=mesh, output_path=output_path)
        return

    if mesh.uv_texture_map is not None:
        _save_mesh_uv_texture_map(mesh=mesh, output_path=output_path)
        return

    _save_mesh_geometry_only(mesh=mesh, output_path=output_path)


def _save_mesh_geometry_only(mesh: "Mesh", output_path: Union[str, Path]) -> None:
    """Save one geometry-only mesh as OBJ or PLY.

    Args:
        mesh: `Mesh` instance with geometry only.
        output_path: Output mesh filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

        assert isinstance(mesh, Mesh), (
            "Expected geometry-only mesh saving input to be a `Mesh` instance. "
            f"{type(mesh)=}"
        )
        assert mesh.vertex_color is None, (
            "Expected geometry-only mesh saving to receive no `vertex_color`. "
            f"{mesh.vertex_color is None=}"
        )
        assert mesh.uv_texture_map is None, (
            "Expected geometry-only mesh saving to receive no `uv_texture_map`. "
            f"{mesh.uv_texture_map is None=}"
        )
        assert mesh.vertex_uv is None, (
            "Expected geometry-only mesh saving to receive no `vertex_uv`. "
            f"{mesh.vertex_uv is None=}"
        )
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    output_mesh_path = _resolve_output_non_uv_mesh_path(output_path=output_path)

    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    vertices_np = mesh.vertices.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()

    if output_mesh_path.suffix.lower() == ".obj":
        with output_mesh_path.open("w", encoding="utf-8") as handle:
            for vertex_row in vertices_np:
                handle.write(
                    "v {:.6f} {:.6f} {:.6f}\n".format(
                        float(vertex_row[0]),
                        float(vertex_row[1]),
                        float(vertex_row[2]),
                    )
                )
            for face_row in faces_np:
                handle.write(
                    "f {} {} {}\n".format(
                        int(face_row[0]) + 1,
                        int(face_row[1]) + 1,
                        int(face_row[2]) + 1,
                    )
                )
        return

    assert output_mesh_path.suffix.lower() == ".ply", (
        "Expected geometry-only mesh saving to resolve to `.obj` or `.ply`. "
        f"{output_mesh_path=}"
    )
    with output_mesh_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices_np.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write(f"element face {faces_np.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex_row in vertices_np:
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


def _save_mesh_vertex_color(mesh: "Mesh", output_path: Union[str, Path]) -> None:
    """Save one vertex-colored mesh as OBJ or PLY.

    Args:
        mesh: `Mesh` instance with vertex-color attributes.
        output_path: Output OBJ/PLY filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

        assert isinstance(mesh, Mesh), (
            "Expected vertex-color mesh saving input to be a `Mesh` instance. "
            f"{type(mesh)=}"
        )
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Path:
        return _resolve_output_non_uv_mesh_path(output_path=output_path)

    output_mesh_path = _normalize_inputs()

    output_mesh_path.parent.mkdir(parents=True, exist_ok=True)

    vertices_cpu = mesh.vertices.detach().cpu()
    faces_cpu = mesh.faces.detach().cpu()
    vertices_np = vertices_cpu.numpy()
    faces_np = faces_cpu.numpy()
    if output_mesh_path.suffix.lower() == ".obj":
        colors_np = _normalize_vertex_color_for_obj(
            vertex_color=mesh.vertex_color
        ).numpy()
        with output_mesh_path.open("w", encoding="utf-8") as handle:
            for vertex_row, color_row in zip(vertices_np, colors_np, strict=True):
                handle.write(
                    "v {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        float(vertex_row[0]),
                        float(vertex_row[1]),
                        float(vertex_row[2]),
                        float(color_row[0]),
                        float(color_row[1]),
                        float(color_row[2]),
                    )
                )
            for face_row in faces_np:
                handle.write(
                    "f {} {} {}\n".format(
                        int(face_row[0]) + 1,
                        int(face_row[1]) + 1,
                        int(face_row[2]) + 1,
                    )
                )
        return

    assert output_mesh_path.suffix.lower() == ".ply", (
        "Expected vertex-color mesh saving to resolve to `.obj` or `.ply`. "
        f"{output_mesh_path=}"
    )
    colors_uint8_np = _normalize_vertex_color_for_ply(
        vertex_color=mesh.vertex_color
    ).numpy()
    with output_mesh_path.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {vertices_np.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write(f"element face {faces_np.shape[0]}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex_row, color_row in zip(vertices_np, colors_uint8_np, strict=True):
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


def _save_mesh_uv_texture_map(mesh: "Mesh", output_path: Union[str, Path]) -> None:
    """Save one UV-textured mesh as OBJ/MTL/PNG assets.

    Args:
        mesh: `Mesh` instance with UV-texture attributes.
        output_path: Output OBJ filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        from data.structures.three_d.mesh.mesh import Mesh

        assert isinstance(mesh, Mesh), (
            "Expected UV-textured mesh saving input to be a `Mesh` instance. "
            f"{type(mesh)=}"
        )
        assert mesh.uv_texture_map is not None, (
            "Expected UV-textured mesh saving to receive `uv_texture_map`. "
            f"{mesh.uv_texture_map is not None=}"
        )
        validate_mesh_uv_convention(convention=mesh.convention)
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    def _normalize_inputs() -> Path:
        return _resolve_output_obj_path(output_path=output_path)

    output_obj_path = _normalize_inputs()

    output_obj_path.parent.mkdir(parents=True, exist_ok=True)

    output_mtl_path = output_obj_path.with_suffix(".mtl")
    output_texture_path = output_obj_path.with_name(
        f"{output_obj_path.stem}_texture.png"
    )

    texture_uint8 = _normalize_uv_texture_map_for_png(
        uv_texture_map=mesh.uv_texture_map
    )
    Image.fromarray(texture_uint8).save(str(output_texture_path))

    with output_mtl_path.open("w", encoding="utf-8") as handle:
        handle.write("newmtl material0\n")
        handle.write("Ka 0.000000 0.000000 0.000000\n")
        handle.write("Kd 1.000000 1.000000 1.000000\n")
        handle.write("Ks 0.000000 0.000000 0.000000\n")
        handle.write("d 1.000000\n")
        handle.write("illum 1\n")
        handle.write(f"map_Kd {output_texture_path.name}\n")

    vertices_np = mesh.vertices.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    vertex_uv_obj = transform_vertex_uv_convention(
        vertex_uv=mesh.vertex_uv.detach().cpu(),
        source_convention=mesh.convention,
        target_convention="obj",
    )
    vertex_uv_np = vertex_uv_obj.numpy()
    face_uvs_np = mesh.face_uvs.detach().cpu().numpy()

    with output_obj_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mtllib {output_mtl_path.name}\n")
        handle.write("usemtl material0\n")
        for vertex_row in vertices_np:
            handle.write(
                "v {:.6f} {:.6f} {:.6f}\n".format(
                    float(vertex_row[0]),
                    float(vertex_row[1]),
                    float(vertex_row[2]),
                )
            )
        for uv_row in vertex_uv_np:
            handle.write(
                "vt {:.6f} {:.6f}\n".format(
                    float(uv_row[0]),
                    float(uv_row[1]),
                )
            )
        for face_row, face_uv_row in zip(faces_np, face_uvs_np, strict=True):
            handle.write(
                "f {}/{} {}/{} {}/{}\n".format(
                    int(face_row[0]) + 1,
                    int(face_uv_row[0]) + 1,
                    int(face_row[1]) + 1,
                    int(face_uv_row[1]) + 1,
                    int(face_row[2]) + 1,
                    int(face_uv_row[2]) + 1,
                )
            )


def _resolve_output_obj_path(output_path: Union[str, Path]) -> Path:
    """Resolve one user output path to a concrete OBJ output path.

    Args:
        output_path: Output OBJ filepath or output directory path.

    Returns:
        Concrete OBJ output filepath.
    """

    assert isinstance(output_path, (str, Path)), (
        "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
    )

    candidate_path = Path(output_path)
    if candidate_path.suffix.lower() == ".obj":
        return candidate_path
    if candidate_path.suffix != "":
        raise AssertionError(
            "Expected mesh saving to target either an `.obj` file or a directory-like "
            "path without a suffix. "
            f"{candidate_path=}"
        )
    return candidate_path / "mesh.obj"


def _resolve_output_non_uv_mesh_path(output_path: Union[str, Path]) -> Path:
    """Resolve one non-UV output path to OBJ or PLY.

    Args:
        output_path: Output mesh filepath or output directory path.

    Returns:
        Concrete non-UV mesh output filepath.
    """

    assert isinstance(output_path, (str, Path)), (
        "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
    )

    candidate_path = Path(output_path)
    if candidate_path.suffix.lower() in (".obj", ".ply"):
        return candidate_path
    if candidate_path.suffix != "":
        raise AssertionError(
            "Expected non-UV mesh saving to target `.obj`, `.ply`, or a "
            "directory-like path without a suffix. "
            f"{candidate_path=}"
        )
    return candidate_path / "mesh.obj"


def _normalize_vertex_color_for_obj(vertex_color: torch.Tensor) -> torch.Tensor:
    """Normalize one vertex-color tensor to float RGB values for OBJ export.

    Args:
        vertex_color: Vertex-color tensor in uint8 or float32 form.

    Returns:
        Float32 vertex-color tensor in `[0, 1]`.
    """

    assert isinstance(vertex_color, torch.Tensor), (
        "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(vertex_color)=}"
    )

    if vertex_color.dtype == torch.uint8:
        return vertex_color.to(dtype=torch.float32).div(255.0).contiguous()
    assert vertex_color.dtype == torch.float32, (
        "Expected vertex-color OBJ export to receive uint8 or float32 input. "
        f"{vertex_color.dtype=}"
    )
    return vertex_color.contiguous()


def _normalize_vertex_color_for_ply(vertex_color: torch.Tensor) -> torch.Tensor:
    """Normalize one vertex-color tensor to uint8 RGB values for PLY export.

    Args:
        vertex_color: Vertex-color tensor in uint8 or float32 form.

    Returns:
        Uint8 vertex-color tensor in `[0, 255]`.
    """

    assert isinstance(vertex_color, torch.Tensor), (
        "Expected `vertex_color` to be a `torch.Tensor`. " f"{type(vertex_color)=}"
    )
    if vertex_color.dtype == torch.uint8:
        return vertex_color.contiguous()
    assert vertex_color.dtype == torch.float32, (
        "Expected vertex-color PLY export to receive uint8 or float32 input. "
        f"{vertex_color.dtype=}"
    )
    return vertex_color.mul(255.0).round().to(dtype=torch.uint8).contiguous()


def _normalize_uv_texture_map_for_png(uv_texture_map: torch.Tensor) -> np.ndarray:
    """Normalize one UV texture tensor to uint8 HWC PNG data.

    Args:
        uv_texture_map: UV texture tensor in HWC layout.

    Returns:
        PNG-ready uint8 HWC array.
    """

    assert isinstance(uv_texture_map, torch.Tensor), (
        "Expected `uv_texture_map` to be a `torch.Tensor`. " f"{type(uv_texture_map)=}"
    )
    assert uv_texture_map.ndim == 3, (
        "Expected UV texture saving to receive HWC tensor layout. "
        f"{uv_texture_map.shape=}"
    )
    assert uv_texture_map.shape[2] == 3, (
        "Expected UV texture saving to receive RGB channels in the last dimension. "
        f"{uv_texture_map.shape=}"
    )

    texture_cpu = uv_texture_map.detach().cpu()
    if texture_cpu.dtype == torch.uint8:
        return texture_cpu.numpy()
    assert texture_cpu.dtype == torch.float32, (
        "Expected UV texture saving to receive uint8 or float32 input. "
        f"{texture_cpu.dtype=}"
    )
    return texture_cpu.mul(255.0).round().to(dtype=torch.uint8).numpy()
