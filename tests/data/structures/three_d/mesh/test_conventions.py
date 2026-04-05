import sys
import types
from pathlib import Path

import pytest
import torch


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

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.save import save_mesh
from data.structures.three_d.mesh.validate import validate_mesh_uv_convention


def _build_uv_mesh(convention: str) -> Mesh:
    """Build one UV-textured mesh for convention tests.

    Args:
        convention: UV-origin convention for the mesh.

    Returns:
        UV-textured mesh.
    """

    assert convention in ("obj", "top_left"), f"{convention=}"
    vertex_uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    if convention == "top_left":
        vertex_uv = vertex_uv.clone()
        vertex_uv[:, 1] = 1.0 - vertex_uv[:, 1]
    return Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        uv_texture_map=torch.tensor(
            [
                [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        ),
        vertex_uv=vertex_uv,
        face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention=convention,
    )


@pytest.mark.parametrize("convention", ["obj", "top_left"])
def test_validate_mesh_uv_convention_accepts_supported_values(
    convention: str,
) -> None:
    validated = validate_mesh_uv_convention(convention=convention)
    assert validated == convention, f"{validated=}"


def test_mesh_requires_convention_when_uv_coordinates_are_provided() -> None:
    with pytest.raises(AssertionError, match="convention"):
        Mesh(
            vertices=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
            ),
            faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            vertex_uv=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            face_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        )


def test_mesh_uv_conversion_round_trip_preserves_texture_and_faces() -> None:
    uv_mesh_obj = _build_uv_mesh(convention="obj")

    uv_mesh_top_left = uv_mesh_obj.to(convention="top_left")
    round_trip_mesh = uv_mesh_top_left.to(convention="obj")

    assert uv_mesh_top_left.convention == "top_left", f"{uv_mesh_top_left.convention=}"
    assert torch.allclose(
        uv_mesh_top_left.vertex_uv,
        torch.tensor(
            [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        atol=1.0e-06,
        rtol=0.0,
    ), f"{uv_mesh_top_left.vertex_uv=}"
    assert torch.equal(
        uv_mesh_top_left.face_uvs, uv_mesh_obj.face_uvs
    ), f"{uv_mesh_top_left.face_uvs=} {uv_mesh_obj.face_uvs=}"
    assert torch.equal(
        uv_mesh_top_left.uv_texture_map, uv_mesh_obj.uv_texture_map
    ), f"{uv_mesh_top_left.uv_texture_map=} {uv_mesh_obj.uv_texture_map=}"
    assert torch.allclose(
        round_trip_mesh.vertex_uv,
        uv_mesh_obj.vertex_uv,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{round_trip_mesh.vertex_uv=} {uv_mesh_obj.vertex_uv=}"


def test_mesh_save_converts_top_left_uvs_to_obj_round_trip(tmp_path: Path) -> None:
    uv_mesh_top_left = _build_uv_mesh(convention="top_left")
    output_obj_path = tmp_path / "face.obj"

    uv_mesh_top_left.save(path=output_obj_path)
    reloaded_mesh = Mesh.load(path=output_obj_path)

    assert reloaded_mesh.convention == "obj", f"{reloaded_mesh.convention=}"
    assert torch.allclose(
        reloaded_mesh.vertex_uv,
        uv_mesh_top_left.to(convention="obj").vertex_uv,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{reloaded_mesh.vertex_uv=}"
    assert torch.equal(
        reloaded_mesh.face_uvs,
        uv_mesh_top_left.face_uvs,
    ), f"{reloaded_mesh.face_uvs=} {uv_mesh_top_left.face_uvs=}"


def test_geometry_only_mesh_save_and_load_round_trip(tmp_path: Path) -> None:
    geometry_only_mesh = Mesh(
        vertices=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64),
    )
    output_obj_path = tmp_path / "geometry.obj"

    geometry_only_mesh.save(path=output_obj_path)
    reloaded_mesh = Mesh.load(path=output_obj_path)

    assert reloaded_mesh.vertex_color is None, f"{reloaded_mesh.vertex_color=}"
    assert reloaded_mesh.uv_texture_map is None, f"{reloaded_mesh.uv_texture_map=}"
    assert reloaded_mesh.vertex_uv is None, f"{reloaded_mesh.vertex_uv=}"
    assert reloaded_mesh.face_uvs is None, f"{reloaded_mesh.face_uvs=}"
    assert reloaded_mesh.convention is None, f"{reloaded_mesh.convention=}"
    assert torch.allclose(
        reloaded_mesh.vertices,
        geometry_only_mesh.vertices,
        atol=1.0e-06,
        rtol=0.0,
    ), f"{reloaded_mesh.vertices=} {geometry_only_mesh.vertices=}"
    assert torch.equal(
        reloaded_mesh.faces,
        geometry_only_mesh.faces,
    ), f"{reloaded_mesh.faces=} {geometry_only_mesh.faces=}"


def test_vertex_color_mesh_save_as_ply_writes_rgb_properties(tmp_path: Path) -> None:
    vertex_color_mesh = Mesh(
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
    output_ply_path = tmp_path / "vertex_color.ply"

    vertex_color_mesh.save(path=output_ply_path)
    ply_text = output_ply_path.read_text(encoding="utf-8")

    assert "property uchar red" in ply_text, f"{ply_text=}"
    assert "property uchar green" in ply_text, f"{ply_text=}"
    assert "property uchar blue" in ply_text, f"{ply_text=}"
    assert "0.000000 0.000000 0.000000 255 0 0" in ply_text, f"{ply_text=}"
    assert "1.000000 0.000000 0.000000 0 128 0" in ply_text, f"{ply_text=}"
    assert "0.000000 1.000000 0.000000 0 0 255" in ply_text, f"{ply_text=}"


def test_save_mesh_rejects_mesh_like_instances(tmp_path: Path) -> None:
    class _MeshLike:
        def __init__(self) -> None:
            self.vertices = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
            self.faces = torch.tensor([[0, 0, 0]], dtype=torch.int64)
            self.vertex_color = None
            self.uv_texture_map = None
            self.vertex_uv = None
            self.face_uvs = None
            self.convention = None

    with pytest.raises(AssertionError, match="Mesh"):
        save_mesh(mesh=_MeshLike(), output_path=tmp_path / "mesh.obj")
