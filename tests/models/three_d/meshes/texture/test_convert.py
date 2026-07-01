"""Tests for shared mesh-texture conversion helpers."""

import pytest
import torch

from models.three_d.meshes.texture.convert import (
    CANONICAL_BFM_VERTEX_UV_PATH,
    build_canonical_bfm_verts_uvs,
)


@pytest.mark.skipif(
    not CANONICAL_BFM_VERTEX_UV_PATH.exists(),
    reason="canonical BFM UV asset (project/submodules/HRN) not present",
)
def test_build_canonical_bfm_verts_uvs_matches_bfm_topology() -> None:
    """The canonical BFM UV builder should return one UV pair per BFM vertex.

    Args:
        None.

    Returns:
        None.
    """

    mean_shape = torch.zeros((35709, 3), dtype=torch.float32)

    verts_uvs = build_canonical_bfm_verts_uvs(mean_shape=mean_shape)

    assert verts_uvs.shape == (35709, 2), f"{verts_uvs.shape=}"
    assert verts_uvs.dtype == torch.float32, f"{verts_uvs.dtype=}"
    assert verts_uvs.device == mean_shape.device, (
        "Expected the canonical UV builder to preserve the requested device. "
        f"{verts_uvs.device=} {mean_shape.device=}"
    )
    assert float(verts_uvs.min().item()) >= 0.0, f"{float(verts_uvs.min().item())=}"
    assert float(verts_uvs.max().item()) <= 1.0, f"{float(verts_uvs.max().item())=}"


@pytest.mark.skipif(
    not CANONICAL_BFM_VERTEX_UV_PATH.exists(),
    reason="canonical BFM UV asset (project/submodules/HRN) not present",
)
def test_build_canonical_bfm_verts_uvs_rejects_vertex_count_mismatch() -> None:
    """The canonical BFM UV builder should reject non-BFM vertex counts.

    Args:
        None.

    Returns:
        None.
    """

    mean_shape = torch.zeros((3, 3), dtype=torch.float32)

    with pytest.raises(AssertionError, match="vertex count"):
        build_canonical_bfm_verts_uvs(mean_shape=mean_shape)
