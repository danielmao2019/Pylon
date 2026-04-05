"""Tests for shared mesh-texture conversion helpers."""

import pytest
import torch

from models.three_d.meshes.texture.convert import build_canonical_bfm_vertex_uv


def test_build_canonical_bfm_vertex_uv_matches_bfm_topology() -> None:
    """The canonical BFM UV builder should return one UV pair per BFM vertex.

    Args:
        None.

    Returns:
        None.
    """

    mean_shape = torch.zeros((35709, 3), dtype=torch.float32)

    vertex_uv = build_canonical_bfm_vertex_uv(mean_shape=mean_shape)

    assert vertex_uv.shape == (35709, 2), f"{vertex_uv.shape=}"
    assert vertex_uv.dtype == torch.float32, f"{vertex_uv.dtype=}"
    assert vertex_uv.device == mean_shape.device, (
        "Expected the canonical UV builder to preserve the requested device. "
        f"{vertex_uv.device=} {mean_shape.device=}"
    )
    assert float(vertex_uv.min().item()) >= 0.0, f"{float(vertex_uv.min().item())=}"
    assert float(vertex_uv.max().item()) <= 1.0, f"{float(vertex_uv.max().item())=}"


def test_build_canonical_bfm_vertex_uv_rejects_vertex_count_mismatch() -> None:
    """The canonical BFM UV builder should reject non-BFM vertex counts.

    Args:
        None.

    Returns:
        None.
    """

    mean_shape = torch.zeros((3, 3), dtype=torch.float32)

    with pytest.raises(AssertionError, match="vertex count"):
        build_canonical_bfm_vertex_uv(mean_shape=mean_shape)
