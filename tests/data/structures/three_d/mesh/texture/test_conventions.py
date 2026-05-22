"""Unit tests for the UV-origin convention transform."""

import sys
import types
from pathlib import Path

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

from data.structures.three_d.mesh.texture.conventions import (
    transform_vertex_uv_convention,
)


def test_identity_when_conventions_match() -> None:
    """Return the UV table unchanged when the conventions are equal.

    Args:
        None.

    Returns:
        None.
    """

    vertex_uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]],
        dtype=torch.float32,
    )

    transformed = transform_vertex_uv_convention(
        vertex_uv=vertex_uv,
        source_convention="obj",
        target_convention="obj",
    )

    assert transformed is vertex_uv, f"{transformed=} {vertex_uv=}"


def test_flips_v_axis_when_conventions_differ() -> None:
    """Flip the V axis (v -> 1 - v) when the conventions differ.

    Args:
        None.

    Returns:
        None.
    """

    vertex_uv = torch.tensor(
        [[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]],
        dtype=torch.float32,
    )

    transformed = transform_vertex_uv_convention(
        vertex_uv=vertex_uv,
        source_convention="obj",
        target_convention="top_left",
    )

    assert torch.allclose(
        transformed,
        torch.tensor(
            [[0.0, 1.0], [1.0, 0.75], [0.5, 0.0]],
            dtype=torch.float32,
        ),
        atol=1.0e-06,
        rtol=0.0,
    ), f"{transformed=}"
    assert torch.equal(
        transformed[:, 0], vertex_uv[:, 0]
    ), f"{transformed[:, 0]=} {vertex_uv[:, 0]=}"
