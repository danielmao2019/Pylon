"""Unit tests for the vertex-color mesh-texture representation."""

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

from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)


def test_normalizes_uint8_to_float01() -> None:
    """Normalize a uint8 vertex_color into contiguous float32 [V,3] in [0,1].

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureVertexColor(
        vertex_color=torch.tensor(
            [[255, 0, 0], [0, 128, 0], [0, 0, 255]],
            dtype=torch.uint8,
        )
    )

    assert texture.vertex_color.dtype == torch.float32, f"{texture.vertex_color.dtype=}"
    assert texture.vertex_color.shape == (3, 3), f"{texture.vertex_color.shape=}"
    assert (
        texture.vertex_color.is_contiguous()
    ), f"{texture.vertex_color.is_contiguous()=}"
    assert torch.allclose(
        texture.vertex_color,
        torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 128.0 / 255.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        atol=1.0e-06,
        rtol=0.0,
    ), f"{texture.vertex_color=}"


def test_rejects_out_of_range_float() -> None:
    """Reject a float32 vertex_color carrying values outside [0,1].

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError, match="at most 1"):
        MeshTextureVertexColor(
            vertex_color=torch.tensor(
                [[1.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
                dtype=torch.float32,
            )
        )


def test_to_rejects_non_none_convention() -> None:
    """Raise when `to` is given a non-None convention.

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureVertexColor(
        vertex_color=torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
    )

    with pytest.raises(AssertionError, match="convention"):
        texture.to(verts_uvs_convention="obj")
