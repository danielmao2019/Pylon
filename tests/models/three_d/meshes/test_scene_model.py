"""Tests for mesh scene-model loading through the data-layer mesh boundary."""

from pathlib import Path

import torch
from torch.testing import assert_close

from data.structures.three_d.mesh.mesh import Mesh
from models.three_d.meshes.scene_model import BaseMeshesSceneModel


class _DummyMeshesSceneModel(BaseMeshesSceneModel):
    """Minimal scene-model subclass for mesh-loading tests.

    Args:
        None.

    Returns:
        None.
    """

    @staticmethod
    def parse_scene_path(path: str) -> str:
        """Return the mesh path unchanged.

        Args:
            path: Input scene path.

        Returns:
            Same path string.
        """

        assert isinstance(path, str), f"{type(path)=}"
        return path

    @staticmethod
    def extract_scene_name(resolved_path: str) -> str:
        """Derive a simple scene name from the resolved path.

        Args:
            resolved_path: Resolved scene path.

        Returns:
            Final path component.
        """

        assert isinstance(resolved_path, str), f"{type(resolved_path)=}"
        return Path(resolved_path).name

    @staticmethod
    def infer_data_dir(resolved_path: str) -> str:
        """Use the mesh-root path itself as the data directory.

        Args:
            resolved_path: Resolved scene path.

        Returns:
            Data-directory path string.
        """

        assert isinstance(resolved_path, str), f"{type(resolved_path)=}"
        return resolved_path


def test_base_meshes_scene_model_loads_through_data_mesh_boundary(
    tmp_path: Path,
) -> None:
    """Load one mesh scene model as one repo `Mesh`.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """

    mesh_root = tmp_path / "mesh_scene"
    mesh_root.mkdir(parents=True, exist_ok=True)
    textured_mesh = Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        uv_texture_map=torch.tensor(
            [[[1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        vertex_uv=torch.tensor(
            [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]],
            dtype=torch.float32,
        ),
        face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    )
    textured_mesh.save(path=mesh_root / "mesh.obj")

    scene_model = _DummyMeshesSceneModel(
        scene_path=str(mesh_root),
        data_dir=mesh_root,
        device=torch.device("cpu"),
    )

    loaded_mesh = scene_model.model
    expected_mesh = Mesh.load(path=mesh_root)

    assert isinstance(loaded_mesh, Mesh), f"{type(loaded_mesh)=}"
    assert_close(loaded_mesh.vertices, expected_mesh.vertices)
    assert torch.equal(
        loaded_mesh.faces, expected_mesh.faces
    ), f"{loaded_mesh.faces=} {expected_mesh.faces=}"
    assert_close(loaded_mesh.uv_texture_map, expected_mesh.uv_texture_map)
    assert_close(loaded_mesh.vertex_uv, expected_mesh.vertex_uv)
    assert torch.equal(
        loaded_mesh.face_uvs, expected_mesh.face_uvs
    ), f"{loaded_mesh.face_uvs=} {expected_mesh.face_uvs=}"


def test_base_meshes_scene_model_to_moves_repo_mesh(
    tmp_path: Path,
) -> None:
    """Move the stored repo mesh through the scene-model `to(...)` boundary.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        None.
    """

    mesh_root = tmp_path / "mesh_scene_to"
    mesh_root.mkdir(parents=True, exist_ok=True)
    Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        uv_texture_map=torch.tensor(
            [[[1.0, 0.0, 0.0]]],
            dtype=torch.float32,
        ),
        vertex_uv=torch.tensor(
            [[0.10, 0.20], [0.30, 0.40], [0.50, 0.60]],
            dtype=torch.float32,
        ),
        face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    ).save(path=mesh_root / "mesh.obj")

    scene_model = _DummyMeshesSceneModel(
        scene_path=str(mesh_root),
        data_dir=mesh_root,
        device=torch.device("cpu"),
    )

    moved_scene_model = scene_model.to(device=torch.device("cpu"))
    assert moved_scene_model is scene_model
    assert isinstance(
        moved_scene_model.model, Mesh
    ), f"{type(moved_scene_model.model)=}"
    assert (
        moved_scene_model.model.device.type == "cpu"
    ), f"{moved_scene_model.model.device=}"
