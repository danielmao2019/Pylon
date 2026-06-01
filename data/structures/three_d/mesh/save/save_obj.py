from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.canonicalize import (
    collapse_seam_shifted_uv_rows,
)
from data.structures.three_d.mesh.texture.conventions import (
    transform_verts_uvs_convention,
)
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def save_obj_mesh(mesh: Mesh, output_path: Union[str, Path]) -> None:
    """Write a Mesh to OBJ, dispatched to the texture-representation-specific writer.

    Args:
        mesh: `Mesh` instance to save.
        output_path: Output OBJ filepath or output directory path.

    Returns:
        None.
    """

    def _validate_inputs() -> None:
        assert isinstance(mesh, Mesh), (
            "Expected `mesh` to be a `Mesh` instance. " f"{type(mesh)=}"
        )

    _validate_inputs()

    obj_path = _resolve_output_obj_path(output_path=output_path)
    obj_path.parent.mkdir(parents=True, exist_ok=True)

    if mesh.texture is None:
        _save_geometry_only_obj(mesh=mesh, obj_path=obj_path)
        return
    if isinstance(mesh.texture, MeshTextureVertexColor):
        _save_vertex_color_obj(mesh=mesh, obj_path=obj_path)
        return
    if isinstance(mesh.texture, MeshTextureUVTextureMap):
        _save_uv_texture_map_obj(mesh=mesh, obj_path=obj_path)
        return
    assert 0, (
        "should not reach here: a Mesh texture is None, MeshTextureVertexColor, "
        f"or MeshTextureUVTextureMap. {type(mesh.texture)=}"
    )


def _save_geometry_only_obj(mesh: Mesh, obj_path: Path) -> None:
    """Write the OBJ v / f lines for a geometry-only mesh.

    Args:
        mesh: Geometry-only `Mesh`.
        obj_path: Concrete OBJ output filepath.

    Returns:
        None.
    """

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    with obj_path.open("w", encoding="utf-8") as handle:
        for vertex_row in verts_np:
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


def _save_vertex_color_obj(mesh: Mesh, obj_path: Path) -> None:
    """Write the OBJ v-x-y-z-r-g-b / f lines for a vertex-colored mesh.

    Args:
        mesh: `Mesh` carrying a `MeshTextureVertexColor`.
        obj_path: Concrete OBJ output filepath.

    Returns:
        None.
    """

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    colors_np = _normalize_vertex_color_for_obj(
        vertex_color=mesh.texture.vertex_color
    ).numpy()
    with obj_path.open("w", encoding="utf-8") as handle:
        for vertex_row, color_row in zip(verts_np, colors_np, strict=True):
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


def _save_uv_texture_map_obj(mesh: Mesh, obj_path: Path) -> None:
    """Write the OBJ plus a sibling MTL and texture PNG for a UV-textured mesh.

    Args:
        mesh: `Mesh` carrying a `MeshTextureUVTextureMap`.
        obj_path: Concrete OBJ output filepath.

    Returns:
        None.
    """

    output_mtl_path = obj_path.with_suffix(".mtl")
    output_texture_path = obj_path.with_name(f"{obj_path.stem}_texture.png")

    texture_uint8 = _normalize_uv_texture_map_for_png(
        uv_texture_map=mesh.texture.uv_texture_map
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

    verts_np = mesh.verts.detach().cpu().numpy()
    faces_np = mesh.faces.detach().cpu().numpy()
    obj_convention_verts_uvs = transform_verts_uvs_convention(
        verts_uvs=mesh.texture.verts_uvs.detach().cpu(),
        source_convention=mesh.texture.convention,
        target_convention="obj",
    )
    obj_verts_uvs, obj_faces_uvs = collapse_seam_shifted_uv_rows(
        verts_uvs=obj_convention_verts_uvs,
        faces_uvs=mesh.texture.faces_uvs.detach().cpu(),
    )
    verts_uvs_np = obj_verts_uvs.numpy()
    faces_uvs_np = obj_faces_uvs.numpy()

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write(f"mtllib {output_mtl_path.name}\n")
        handle.write("usemtl material0\n")
        for vertex_row in verts_np:
            handle.write(
                "v {:.6f} {:.6f} {:.6f}\n".format(
                    float(vertex_row[0]),
                    float(vertex_row[1]),
                    float(vertex_row[2]),
                )
            )
        for uv_row in verts_uvs_np:
            handle.write(
                "vt {:.6f} {:.6f}\n".format(float(uv_row[0]), float(uv_row[1]))
            )
        for face_row, face_uv_row in zip(faces_np, faces_uvs_np, strict=True):
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
    """Resolve an output path to a concrete `.obj` file path.

    Args:
        output_path: Output OBJ filepath or output directory path.

    Returns:
        Concrete OBJ output filepath (an `.obj` path, or `<dir>/mesh.obj`).
    """

    def _validate_inputs() -> None:
        assert isinstance(output_path, (str, Path)), (
            "Expected `output_path` to be a `str` or `Path`. " f"{type(output_path)=}"
        )

    _validate_inputs()

    candidate_path = Path(output_path)
    if candidate_path.suffix.lower() == ".obj":
        return candidate_path
    assert candidate_path.suffix == "", (
        "Expected OBJ mesh saving to target an `.obj` file or a directory-like "
        "path without a suffix. "
        f"{candidate_path=}"
    )
    return candidate_path / "mesh.obj"


def _normalize_vertex_color_for_obj(vertex_color: torch.Tensor) -> torch.Tensor:
    """Convert one vertex-color tensor to float32 RGB `[0, 1]` for OBJ export.

    Args:
        vertex_color: Vertex-color tensor in uint8 `[0, 255]` or float32
            `[0, 1]` form.

    Returns:
        Float32 vertex-color tensor in `[0, 1]`.
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
        return vertex_color.to(dtype=torch.float32).div(255.0).contiguous()
    return vertex_color.contiguous()


def _normalize_uv_texture_map_for_png(uv_texture_map: torch.Tensor) -> np.ndarray:
    """Convert one UV texture map to a uint8 HWC array for PNG export.

    Args:
        uv_texture_map: UV texture map tensor in HWC RGB layout.

    Returns:
        PNG-ready uint8 HWC array.
    """

    def _validate_inputs() -> None:
        assert isinstance(uv_texture_map, torch.Tensor), (
            "Expected `uv_texture_map` to be a `torch.Tensor`. "
            f"{type(uv_texture_map)=}"
        )
        assert uv_texture_map.ndim == 3 and uv_texture_map.shape[2] == 3, (
            "Expected `uv_texture_map` to use HWC RGB layout. "
            f"{uv_texture_map.shape=}"
        )
        assert uv_texture_map.dtype in (torch.uint8, torch.float32), (
            "Expected `uv_texture_map` to be uint8 or float32. "
            f"{uv_texture_map.dtype=}"
        )

    _validate_inputs()

    texture_cpu = uv_texture_map.detach().cpu()
    if texture_cpu.dtype == torch.uint8:
        return texture_cpu.numpy()
    return texture_cpu.mul(255.0).round().to(dtype=torch.uint8).numpy()
