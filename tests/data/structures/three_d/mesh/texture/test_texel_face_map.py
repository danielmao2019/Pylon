"""Unit tests for the texel -> mesh-face correspondence builder."""

import sys
import types
from pathlib import Path

import pytest
import torch


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


REPO_ROOT = Path(__file__).resolve().parents[6]
_install_namespace_package(package_name="data", package_path=REPO_ROOT / "data")
_install_namespace_package(
    package_name="data.structures", package_path=REPO_ROOT / "data" / "structures"
)
_install_namespace_package(
    package_name="data.structures.three_d",
    package_path=REPO_ROOT / "data" / "structures" / "three_d",
)

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)
from data.structures.three_d.mesh.texture.texel_face_map import build_texel_face_map


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="build_texel_face_map uses nvdiffrast's CUDA rasterizer",
)


_CUDA_DEVICE = torch.device("cuda")


def _build_identity_uv_mesh() -> Mesh:
    """Build one identity-UV single-face mesh on CUDA.

    Args:
        None.

    Returns:
        One UV-textured mesh with a single small face entirely inside `[0, 1]`.
    """

    return Mesh(
        verts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=_CUDA_DEVICE,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64, device=_CUDA_DEVICE),
        texture=MeshTextureUVTextureMap(
            uv_texture_map=torch.zeros(
                (1, 1, 3), dtype=torch.float32, device=_CUDA_DEVICE
            ),
            verts_uvs=torch.tensor(
                [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]],
                dtype=torch.float32,
                device=_CUDA_DEVICE,
            ),
            faces_uvs=torch.tensor(
                [[0, 1, 2]], dtype=torch.int64, device=_CUDA_DEVICE
            ),
            convention="obj",
        ),
    )


def test_build_texel_face_map_returns_texel_face_index_and_barycentric() -> None:
    """Return texel_face_index [T, T] int64 and texel_face_barycentric [T, T, 3] float32 with the expected shapes and -1 / NaN sentinels at unoccupied texels.

    Args:
        None.

    Returns:
        None.
    """

    mesh = _build_identity_uv_mesh()

    texel_face_map = build_texel_face_map(mesh=mesh, texture_size=8)

    assert set(texel_face_map.keys()) == {
        "texel_face_index",
        "texel_face_barycentric",
    }, f"{set(texel_face_map.keys())=}"

    texel_face_index = texel_face_map["texel_face_index"]
    texel_face_barycentric = texel_face_map["texel_face_barycentric"]

    assert texel_face_index.shape == (8, 8), f"{texel_face_index.shape=}"
    assert texel_face_index.dtype == torch.int64, f"{texel_face_index.dtype=}"
    assert texel_face_barycentric.shape == (8, 8, 3), f"{texel_face_barycentric.shape=}"
    assert texel_face_barycentric.dtype == torch.float32, (
        f"{texel_face_barycentric.dtype=}"
    )
    assert bool((texel_face_index == -1).any().item()), (
        "Expected at least one unoccupied texel for an identity-UV single-face "
        "mesh at texture_size=8. "
        f"{texel_face_index=}"
    )


def test_build_texel_face_map_maps_identity_face_to_top_row() -> None:
    """Assign face 0 to the top texel rows on an identity-UV face with small-v corners (top_left v-convention is the rasterizer-buffer mapping).

    Args:
        None.

    Returns:
        None.
    """

    mesh = _build_identity_uv_mesh()

    texel_face_map = build_texel_face_map(mesh=mesh, texture_size=8)
    texel_face_index = texel_face_map["texel_face_index"]

    assert bool((texel_face_index[0] == 0).any().item()), (
        "Expected the top texel row to receive face 0 under nvdiffrast's "
        "small-v-to-negative-clip-y mapping. "
        f"{texel_face_index[0]=}"
    )


def test_build_texel_face_map_covers_both_sides_of_cylindrical_seam() -> None:
    """For a seam-safe canonical mesh whose only face spans u in {0.95, 1.05, 1.02}, both the u-near-1 and u-near-0 texel columns get assigned to that face (cylindrical wrap coverage via internal seam-side duplication).

    Args:
        None.

    Returns:
        None.
    """

    mesh = Mesh(
        verts=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=torch.float32,
            device=_CUDA_DEVICE,
        ),
        faces=torch.tensor([[0, 1, 2]], dtype=torch.int64, device=_CUDA_DEVICE),
        texture=MeshTextureUVTextureMap(
            uv_texture_map=torch.zeros(
                (1, 1, 3), dtype=torch.float32, device=_CUDA_DEVICE
            ),
            verts_uvs=torch.tensor(
                [[0.95, 0.20], [1.05, 0.25], [1.02, 0.80]],
                dtype=torch.float32,
                device=_CUDA_DEVICE,
            ),
            faces_uvs=torch.tensor(
                [[0, 1, 2]], dtype=torch.int64, device=_CUDA_DEVICE
            ),
            convention="obj",
        ),
    )

    texel_face_map = build_texel_face_map(mesh=mesh, texture_size=16)
    texel_face_index = texel_face_map["texel_face_index"]

    near_one_column = texel_face_index[:, -1]
    near_zero_column = texel_face_index[:, 0]
    assert bool((near_one_column == 0).any().item()), (
        "Expected the u-near-1 column to receive face 0 (primary copy "
        "rasterizes the right side of the seam). "
        f"{near_one_column=}"
    )
    assert bool((near_zero_column == 0).any().item()), (
        "Expected the u-near-0 column to receive face 0 via the mirror "
        "copy (mirror copy rasterizes the left side of the seam). "
        f"{near_zero_column=}"
    )


def test_build_texel_face_map_barycentric_recovers_face_vertex_attributes() -> None:
    """Barycentric-interpolating the owning face's three corner UVs recovers each occupied texel's own center UV, so a corner-permuted barycentric is caught (not merely an in-range convex combination).

    Args:
        None.

    Returns:
        None.
    """

    mesh = _build_identity_uv_mesh()
    texture_size = 64

    texel_face_map = build_texel_face_map(mesh=mesh, texture_size=texture_size)
    texel_face_index = texel_face_map["texel_face_index"]
    texel_face_barycentric = texel_face_map["texel_face_barycentric"]

    occupied_mask = texel_face_index >= 0
    assert bool(occupied_mask.any().item()), f"{occupied_mask.any()=}"

    corner_uvs = mesh.texture.verts_uvs[
        mesh.texture.faces_uvs[texel_face_index.clamp(min=0)]
    ]
    interpolated_uv = (corner_uvs * texel_face_barycentric.unsqueeze(-1)).sum(dim=-2)

    axis = (
        torch.arange(texture_size, device=_CUDA_DEVICE, dtype=torch.float32) + 0.5
    ) / texture_size
    expected_u = axis.view(1, texture_size).expand(texture_size, texture_size)
    expected_v = axis.view(texture_size, 1).expand(texture_size, texture_size)
    expected_uv = torch.stack([expected_u, expected_v], dim=-1)

    max_error = float(
        (interpolated_uv[occupied_mask] - expected_uv[occupied_mask]).abs().max().item()
    )
    assert max_error < 1.0e-3, (
        "Expected barycentric-interpolated corner UVs to recover each occupied "
        "texel's center UV; a corner-permuted barycentric fails this. "
        f"{max_error=}"
    )
