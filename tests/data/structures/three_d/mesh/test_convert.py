"""Regression tests for the mesh convert-module boundary."""

import sys
import types
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from torch.testing import assert_close


def _install_namespace_package(package_name: str, package_path: Path) -> None:
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
    mesh_from_open3d,
    mesh_from_pytorch3d,
    mesh_from_trimesh,
    mesh_to_pytorch3d,
)


def _build_vertex_color_mesh() -> Mesh:
    """Build one vertex-colored mesh for round-trip tests.

    Args:
        None.

    Returns:
        One CPU-owned vertex-colored repo mesh.
    """

    return Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        vertex_color=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )


def test_mesh_round_trip_through_pytorch3d_preserves_vertex_colors() -> None:
    """Round-trip one vertex-colored mesh through PyTorch3D.

    Args:
        None.

    Returns:
        None.
    """

    mesh = _build_vertex_color_mesh()

    pytorch3d_mesh = mesh_to_pytorch3d(mesh=mesh, device=torch.device("cpu"))
    assert len(pytorch3d_mesh) == 1, f"{len(pytorch3d_mesh)=}"

    round_tripped_mesh = mesh_from_pytorch3d(
        mesh=pytorch3d_mesh,
        convention="obj",
    )
    assert isinstance(round_tripped_mesh, Mesh), f"{type(round_tripped_mesh)=}"
    assert_close(round_tripped_mesh.vertices, mesh.vertices)
    assert torch.equal(
        round_tripped_mesh.faces, mesh.faces
    ), f"{round_tripped_mesh.faces=} {mesh.faces=}"
    assert_close(round_tripped_mesh.vertex_color, mesh.vertex_color)
    assert round_tripped_mesh.vertex_uv is None, f"{round_tripped_mesh.vertex_uv=}"
    assert round_tripped_mesh.face_uvs is None, f"{round_tripped_mesh.face_uvs=}"


def test_mesh_from_open3d_preserves_vertex_colors() -> None:
    """Convert one colored legacy Open3D mesh into the repo mesh type.

    Args:
        None.

    Returns:
        None.
    """

    legacy_mesh = o3d.geometry.TriangleMesh()
    legacy_mesh.vertices = o3d.utility.Vector3dVector(
        np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
    )
    legacy_mesh.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2]], dtype=np.int32)
    )
    legacy_mesh.vertex_colors = o3d.utility.Vector3dVector(
        np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    )

    converted_mesh = mesh_from_open3d(mesh=legacy_mesh)
    assert isinstance(converted_mesh, Mesh), f"{type(converted_mesh)=}"
    assert_close(
        converted_mesh.vertices,
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        converted_mesh.faces,
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
    ), f"{converted_mesh.faces=}"
    assert_close(
        converted_mesh.vertex_color,
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )


def test_mesh_from_trimesh_preserves_opaque_rgba_vertex_colors() -> None:
    """Convert one opaque RGBA trimesh into the repo mesh type.

    Args:
        None.

    Returns:
        None.
    """

    source_mesh = trimesh.Trimesh(
        vertices=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        vertex_colors=np.array(
            [[255, 0, 0, 255], [0, 128, 0, 255], [0, 0, 255, 255]],
            dtype=np.uint8,
        ),
        process=False,
    )

    converted_mesh = mesh_from_trimesh(mesh=source_mesh)
    assert isinstance(converted_mesh, Mesh), f"{type(converted_mesh)=}"
    assert_close(
        converted_mesh.vertices,
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        converted_mesh.faces,
        torch.tensor([[0, 1, 2]], dtype=torch.int64),
    ), f"{converted_mesh.faces=}"
    assert_close(
        converted_mesh.vertex_color,
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 128.0 / 255.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    )
