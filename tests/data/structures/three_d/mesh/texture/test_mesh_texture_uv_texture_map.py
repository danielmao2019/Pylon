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


def _build_seam_safe_verts_uvs() -> torch.Tensor:
    """Build one seam-safe verts_uvs table whose only face has u-span <= 0.5.

    Args:
        None.

    Returns:
        Float32 `[3, 2]` UV-coordinate table.
    """

    return torch.tensor(
        [[0.1, 0.1], [0.4, 0.1], [0.1, 0.4]],
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
            verts_uvs=_build_seam_safe_verts_uvs(),
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
        verts_uvs=_build_seam_safe_verts_uvs(),
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


def test_accepts_seam_safe_verts_uvs_outside_unit_interval() -> None:
    """Accept verts_uvs whose u extends beyond 1.0 when each face is non-wrapping (its largest cyclic gap is the wraparound gap), the seam-safe canonical form.

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureUVTextureMap(
        uv_texture_map=_build_uv_texture_map(),
        verts_uvs=torch.tensor(
            [[0.95, 0.20], [1.05, 0.25], [1.02, 0.80]],
            dtype=torch.float32,
        ),
        faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    )

    assert float(texture.verts_uvs.max().item()) > 1.0, f"{texture.verts_uvs=}"


def test_accepts_wide_non_wrapping_face() -> None:
    """Accept a wide face whose u-span exceeds 0.5 but whose corners are contiguous (largest cyclic gap is the wraparound gap) — a wide face is not a wrapping face.

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureUVTextureMap(
        uv_texture_map=_build_uv_texture_map(),
        verts_uvs=torch.tensor(
            [[0.293, 0.20], [0.735, 0.25], [0.801, 0.80]],
            dtype=torch.float32,
        ),
        faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    )

    face_u = texture.verts_uvs[texture.faces_uvs.to(dtype=torch.int64), 0]
    span = float((face_u.max(dim=1).values - face_u.min(dim=1).values).max().item())
    assert span > 0.5, f"test fixture must be a wide face, {span=}"


def test_rejects_wrapping_face() -> None:
    """Reject a face whose largest cyclic gap is an interior gap (its corners straddle the cylindrical wrap and were not seam-shifted into contiguous canonical form).

    Args:
        None.

    Returns:
        None.
    """

    with pytest.raises(AssertionError, match="non-wrapping"):
        MeshTextureUVTextureMap(
            uv_texture_map=_build_uv_texture_map(),
            verts_uvs=torch.tensor(
                [[0.95, 0.20], [0.05, 0.25], [0.02, 0.80]],
                dtype=torch.float32,
            ),
            faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
            convention="obj",
        )


def test_to_converts_uv_convention() -> None:
    """Return a texture whose verts_uvs is converted to the target convention.

    Args:
        None.

    Returns:
        None.
    """

    texture = MeshTextureUVTextureMap(
        uv_texture_map=_build_uv_texture_map(),
        verts_uvs=_build_seam_safe_verts_uvs(),
        faces_uvs=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        convention="obj",
    )

    converted = texture.to(convention="top_left")

    assert converted.convention == "top_left", f"{converted.convention=}"
    assert torch.allclose(
        converted.verts_uvs,
        torch.tensor(
            [[0.1, 0.9], [0.4, 0.9], [0.1, 0.6]],
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
