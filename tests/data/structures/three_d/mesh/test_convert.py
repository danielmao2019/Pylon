"""Unit tests for the mesh convert-module framework interop boundary."""

import sys
import types
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from torch.testing import assert_close


def _install_namespace_package(package_name: str, package_path: Path) -> None:
    """Install one namespace package so the data tree imports without setup.

    Args:
        package_name: Dotted package name to register.
        package_path: Filesystem directory backing the package.

    Returns:
        None.
    """

    if package_name in sys.modules:
        return

    module = types.ModuleType(package_name)
    module.__file__ = str(package_path / "__init__.py")
    module.__path__ = [str(package_path)]
    sys.modules[package_name] = module


REPO_ROOT = Path(__file__).resolve().parents[5]
_install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
_install_namespace_package(
    package_name="data.structures", package_path=REPO_ROOT / "data" / "structures"
)
_install_namespace_package(
    package_name="data.structures.three_d",
    package_path=REPO_ROOT / "data" / "structures" / "three_d",
)

from data.structures.three_d.mesh import (
    Mesh,
    MeshTextureUVTextureMap,
    MeshTextureVertexColor,
    mesh_from_open3d,
    mesh_from_pytorch3d,
    mesh_from_trimesh,
    mesh_to_open3d,
    mesh_to_pytorch3d,
    mesh_to_trimesh,
)


def _write_seamed_uv_obj(directory: Path) -> Path:
    """Write one seamed UV-textured OBJ with a sibling MTL and texture PNG.

    The OBJ is a unit square (4 distinct positions, 2 triangles) whose shared
    diagonal verts carry different UVs per triangle, so the asset has a UV
    seam: 4 geometry verts but 6 UV coordinates.

    Args:
        directory: Directory in which to write the OBJ / MTL / PNG.

    Returns:
        Path to the written OBJ file.
    """

    obj_path = directory / "seam.obj"
    mtl_path = directory / "seam.mtl"
    texture_path = directory / "seam_texture.png"

    obj_path.write_text(
        "mtllib seam.mtl\n"
        "usemtl material0\n"
        "v 0.0 0.0 0.0\n"
        "v 1.0 0.0 0.0\n"
        "v 1.0 1.0 0.0\n"
        "v 0.0 1.0 0.0\n"
        "vt 0.0 0.0\n"
        "vt 1.0 0.0\n"
        "vt 1.0 1.0\n"
        "vt 0.0 1.0\n"
        "vt 0.0 0.5\n"
        "vt 1.0 0.5\n"
        "f 1/1 2/2 3/3\n"
        "f 1/5 3/6 4/4\n",
        encoding="utf-8",
    )
    mtl_path.write_text(
        "newmtl material0\nmap_Kd seam_texture.png\n",
        encoding="utf-8",
    )
    Image.fromarray(np.full((4, 4, 3), 128, dtype=np.uint8)).save(str(texture_path))
    return obj_path


def _build_vertex_color_mesh() -> Mesh:
    """Build one vertex-colored mesh for round-trip tests.

    Args:
        None.

    Returns:
        One CPU-owned vertex-colored repo mesh.
    """

    return Mesh(
        verts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        texture=MeshTextureVertexColor(
            vertex_color=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        ),
    )


def _build_uv_textured_mesh() -> Mesh:
    """Build one UV-textured mesh on the geometry domain.

    Args:
        None.

    Returns:
        One CPU-owned UV-textured repo mesh.
    """

    return Mesh(
        verts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        texture=MeshTextureUVTextureMap(
            uv_texture_map=torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
                ],
                dtype=torch.float32,
            ),
            verts_uvs=torch.tensor(
                [[0.1, 0.1], [0.4, 0.1], [0.1, 0.4]],
                dtype=torch.float32,
            ),
            faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            convention="obj",
        ),
    )


def test_mesh_from_trimesh_welds_seam_to_geometry_domain(tmp_path: Path) -> None:
    """Weld a per-corner-expanded seamed UV mesh onto the geometry domain.

    A seamed UV mesh that trimesh loads in per-corner-expanded form (V == U)
    must come through `mesh_from_trimesh` on the canonical geometry domain
    (V <= U, distinct positions), with the seam carried only by verts_uvs /
    faces_uvs. Enforces the seam contract (task.md design.2).

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """

    obj_path = _write_seamed_uv_obj(directory=tmp_path)
    source_mesh = trimesh.load(str(obj_path), force="mesh", process=False)
    assert source_mesh.visual.uv is not None, f"{source_mesh.visual.uv=}"
    assert len(source_mesh.vertices) == 6, f"{len(source_mesh.vertices)=}"

    mesh = mesh_from_trimesh(mesh=source_mesh, verts_uvs_convention="obj")
    assert isinstance(mesh, Mesh), f"{type(mesh)=}"
    assert isinstance(mesh.texture, MeshTextureUVTextureMap), f"{type(mesh.texture)=}"

    vertex_count = int(mesh.verts.shape[0])
    uv_count = int(mesh.texture.verts_uvs.shape[0])
    assert vertex_count == 4, f"{vertex_count=}"
    assert uv_count == 6, f"{uv_count=}"
    assert vertex_count <= uv_count, f"{vertex_count=} {uv_count=}"

    unique_positions = torch.unique(mesh.verts, dim=0)
    assert (
        unique_positions.shape[0] == vertex_count
    ), f"{unique_positions.shape=} {vertex_count=}"
    assert (
        mesh.texture.faces_uvs.shape == mesh.faces.shape
    ), f"{mesh.texture.faces_uvs.shape=} {mesh.faces.shape=}"


def test_vertex_count_is_loader_independent(tmp_path: Path) -> None:
    """Make len(mesh.verts) identical across PyTorch3D and trimesh loaders.

    For one OBJ asset, `len(mesh.verts)` must be identical whether the mesh
    is loaded via `Mesh.load` (PyTorch3D) or via `mesh_from_trimesh`, since both
    land on the canonical geometry domain.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        None.
    """

    obj_path = _write_seamed_uv_obj(directory=tmp_path)

    pytorch3d_loaded_mesh = Mesh.load(path=obj_path)
    trimesh_loaded_mesh = mesh_from_trimesh(
        mesh=trimesh.load(str(obj_path), force="mesh", process=False),
        verts_uvs_convention="obj",
    )

    assert int(pytorch3d_loaded_mesh.verts.shape[0]) == int(
        trimesh_loaded_mesh.verts.shape[0]
    ), (f"{pytorch3d_loaded_mesh.verts.shape=} " f"{trimesh_loaded_mesh.verts.shape=}")
    assert (
        int(pytorch3d_loaded_mesh.verts.shape[0]) == 4
    ), f"{pytorch3d_loaded_mesh.verts.shape=}"


def test_trimesh_uv_round_trip_preserves_geometry() -> None:
    """Round-trip one UV-textured mesh through trimesh.

    `mesh_to_trimesh` then `mesh_from_trimesh` must preserve geometry, UV, and
    texture (expand then weld is identity on the geometry domain).

    Args:
        None.

    Returns:
        None.
    """

    mesh = _build_uv_textured_mesh()

    trimesh_mesh = mesh_to_trimesh(mesh=mesh)
    round_tripped_mesh = mesh_from_trimesh(
        mesh=trimesh_mesh, verts_uvs_convention="obj"
    )

    assert isinstance(
        round_tripped_mesh.texture, MeshTextureUVTextureMap
    ), f"{type(round_tripped_mesh.texture)=}"
    assert int(round_tripped_mesh.verts.shape[0]) == int(
        mesh.verts.shape[0]
    ), f"{round_tripped_mesh.verts.shape=} {mesh.verts.shape=}"

    sorted_original = mesh.verts[
        torch.argsort(mesh.verts[:, 0] * 1.0e06 + mesh.verts[:, 1])
    ]
    sorted_round_trip = round_tripped_mesh.verts[
        torch.argsort(
            round_tripped_mesh.verts[:, 0] * 1.0e06 + round_tripped_mesh.verts[:, 1]
        )
    ]
    assert_close(sorted_round_trip, sorted_original)
    assert_close(
        round_tripped_mesh.texture.uv_texture_map,
        mesh.texture.uv_texture_map,
    )

    original_uv_by_face = mesh.texture.verts_uvs[mesh.texture.faces_uvs.reshape(-1)]
    round_trip_uv_by_face = round_tripped_mesh.texture.verts_uvs[
        round_tripped_mesh.texture.faces_uvs.reshape(-1)
    ]
    assert_close(round_trip_uv_by_face, original_uv_by_face)


def test_pytorch3d_round_trip_preserves_texture() -> None:
    """Round-trip vertex-colored and UV-textured meshes through PyTorch3D.

    `mesh_to_pytorch3d` then `mesh_from_pytorch3d` must preserve geometry and
    texture for both vertex-colored and UV-textured meshes.

    Args:
        None.

    Returns:
        None.
    """

    vertex_color_mesh = _build_vertex_color_mesh()
    pytorch3d_vc = mesh_to_pytorch3d(mesh=vertex_color_mesh, device=torch.device("cpu"))
    round_tripped_vc = mesh_from_pytorch3d(
        mesh=pytorch3d_vc, verts_uvs_convention="obj"
    )
    assert isinstance(
        round_tripped_vc.texture, MeshTextureVertexColor
    ), f"{type(round_tripped_vc.texture)=}"
    assert_close(round_tripped_vc.verts, vertex_color_mesh.verts)
    assert torch.equal(
        round_tripped_vc.faces, vertex_color_mesh.faces
    ), f"{round_tripped_vc.faces=} {vertex_color_mesh.faces=}"
    assert_close(
        round_tripped_vc.texture.vertex_color,
        vertex_color_mesh.texture.vertex_color,
    )

    uv_mesh = _build_uv_textured_mesh()
    pytorch3d_uv = mesh_to_pytorch3d(mesh=uv_mesh, device=torch.device("cpu"))
    round_tripped_uv = mesh_from_pytorch3d(
        mesh=pytorch3d_uv, verts_uvs_convention="obj"
    )
    assert isinstance(
        round_tripped_uv.texture, MeshTextureUVTextureMap
    ), f"{type(round_tripped_uv.texture)=}"
    assert_close(round_tripped_uv.verts, uv_mesh.verts)
    assert torch.equal(
        round_tripped_uv.faces, uv_mesh.faces
    ), f"{round_tripped_uv.faces=} {uv_mesh.faces=}"
    assert_close(
        round_tripped_uv.texture.uv_texture_map,
        uv_mesh.texture.uv_texture_map,
    )
    assert_close(round_tripped_uv.texture.verts_uvs, uv_mesh.texture.verts_uvs)
    assert torch.equal(
        round_tripped_uv.texture.faces_uvs, uv_mesh.texture.faces_uvs
    ), f"{round_tripped_uv.texture.faces_uvs=} {uv_mesh.texture.faces_uvs=}"


def test_open3d_round_trip_preserves_vertex_color() -> None:
    """Round-trip one vertex-colored mesh through Open3D.

    `mesh_to_open3d` then `mesh_from_open3d` must preserve geometry and vertex
    colors (the Open3D path carries no UV texture).

    Args:
        None.

    Returns:
        None.
    """

    mesh = _build_vertex_color_mesh()

    open3d_mesh = mesh_to_open3d(mesh=mesh)
    round_tripped_mesh = mesh_from_open3d(mesh=open3d_mesh)

    assert isinstance(
        round_tripped_mesh.texture, MeshTextureVertexColor
    ), f"{type(round_tripped_mesh.texture)=}"
    assert_close(round_tripped_mesh.verts, mesh.verts)
    assert torch.equal(
        round_tripped_mesh.faces, mesh.faces
    ), f"{round_tripped_mesh.faces=} {mesh.faces=}"
    assert_close(
        round_tripped_mesh.texture.vertex_color,
        mesh.texture.vertex_color,
    )
