"""Unit tests for the uv-texture-map mesh-texture representation."""

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

from data.structures.three_d.mesh.texture.mesh_texture_uv_texture_map import (
    MeshTextureUVTextureMap,
)


def _build_uv_texture_map() -> torch.Tensor:
    """Build one small float32 HWC uv_texture_map.

    Args:
        None.

    Returns:
        Float32 `[2, 2, 3]` UV texture map.
    """

    return torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ],
        dtype=torch.float32,
    )


def test_rejects_faces_uvs_index_out_of_range() -> None:
    """Reject faces_uvs whose indices do not reference valid verts_uvs rows.

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError, match="verts_uvs"):
        MeshTextureUVTextureMap(
            uv_texture_map=_build_uv_texture_map(),
            verts_uvs=torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                dtype=torch.float32,
            ),
            faces_uvs=torch.tensor([[0, 1, 3]], dtype=torch.int64),
            convention="obj",
        )


def test_normalizes_uint8_texture_map() -> None:
    """Normalize a uint8 uv_texture_map into contiguous float32 HWC in [0,1].

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureUVTextureMap(
        uv_texture_map=torch.tensor(
            [
                [[255, 0, 0], [0, 255, 0]],
                [[0, 0, 255], [255, 255, 0]],
            ],
            dtype=torch.uint8,
        ),
        verts_uvs=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    )

    assert (
        texture.uv_texture_map.dtype == torch.float32
    ), f"{texture.uv_texture_map.dtype=}"
    assert texture.uv_texture_map.shape == (2, 2, 3), f"{texture.uv_texture_map.shape=}"
    assert (
        texture.uv_texture_map.is_contiguous()
    ), f"{texture.uv_texture_map.is_contiguous()=}"
    assert (
        float(texture.uv_texture_map.max().item()) <= 1.0
    ), f"{texture.uv_texture_map.max()=}"
    assert torch.allclose(
        texture.uv_texture_map[0, 0],
        torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        atol=1.0e-06,
        rtol=0.0,
    ), f"{texture.uv_texture_map[0, 0]=}"


def test_to_converts_uv_convention() -> None:
    """Return a texture whose verts_uvs is converted to the target convention.

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureUVTextureMap(
        uv_texture_map=_build_uv_texture_map(),
        verts_uvs=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    )

    converted = texture.to(convention="top_left")

    assert converted.convention == "top_left", f"{converted.convention=}"
    assert torch.allclose(
        converted.verts_uvs,
        torch.tensor(
            [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            dtype=torch.float32,
        ),
        atol=1.0e-06,
        rtol=0.0,
    ), f"{converted.verts_uvs=}"
    assert torch.equal(
        converted.faces_uvs, texture.faces_uvs
    ), f"{converted.faces_uvs=} {texture.faces_uvs=}"
    assert torch.equal(
        converted.uv_texture_map, texture.uv_texture_map
    ), f"{converted.uv_texture_map=} {texture.uv_texture_map=}"
