"""Tests for unit-weighted per-vertex normal computation."""

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from models.three_d.meshes.ops import compute_vertex_normals


def _face_unit_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute a triangle's unit normal using the function-under-test convention.

    The function under test forms each face normal as `cross(v0 - v1, v1 - v2)`,
    so this helper reproduces that exact cross-product order and orientation.

    Args:
        v0: First triangle vertex position, shape `[3]`, dtype float64.
        v1: Second triangle vertex position, shape `[3]`, dtype float64.
        v2: Third triangle vertex position, shape `[3]`, dtype float64.

    Returns:
        The triangle's L2-normalized normal, shape `[3]`, dtype float64.
    """

    assert v0.shape == (3,), f"{v0.shape=}"
    assert v1.shape == (3,), f"{v1.shape=}"
    assert v2.shape == (3,), f"{v2.shape=}"
    raw = np.cross(v0 - v1, v1 - v2)
    return raw / np.linalg.norm(raw)


def test_output_is_unit_length() -> None:
    """Every returned per-vertex normal is L2-normalized on a non-degenerate mesh.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)

    normals = compute_vertex_normals(verts=verts, faces=faces, weights="unit")

    norms = torch.linalg.norm(normals, dim=-1)
    assert_close(norms, torch.ones_like(norms))


def test_single_planar_triangle_orientation() -> None:
    """A single z=0 planar triangle yields the `(0, 0, +1)` unit normal at all verts.

    The function forms each face normal as `cross(v0 - v1, v1 - v2)`; for the
    chosen triangle this convention produces the positive-z orientation, asserted
    here so the reviewer knows the sign the function yields.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    normals = compute_vertex_normals(verts=verts, faces=faces, weights="unit")

    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    assert_close(normals, expected)


def test_unit_weighting_not_area_weighting() -> None:
    """Shared-edge tent verifies unit weighting, not area weighting, of face normals.

    Two triangles share edge A-B with clearly different areas (a non-coplanar
    tent). The shared vertex A's normal must equal `normalize(n0 + n1)` of the two
    UNIT face normals and must differ from the area-weighted
    `normalize(area0 * n0 + area1 * n1)`, since the areas differ. Each apex-only
    vertex carries the unit normal of its single incident face.

    Args:
        None.

    Returns:
        None.
    """

    a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    c = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    d = np.array([0.0, -1.0, 1.0], dtype=np.float64)

    n0 = _face_unit_normal(v0=a, v1=b, v2=c)
    n1 = _face_unit_normal(v0=a, v1=b, v2=d)
    raw0 = np.cross(a - b, b - c)
    raw1 = np.cross(a - b, b - d)
    area0 = 0.5 * np.linalg.norm(raw0)
    area1 = 0.5 * np.linalg.norm(raw1)
    assert not np.isclose(area0, area1), f"{area0=}, {area1=}"

    unit_weighted = n0 + n1
    unit_weighted = unit_weighted / np.linalg.norm(unit_weighted)
    area_weighted = area0 * n0 + area1 * n1
    area_weighted = area_weighted / np.linalg.norm(area_weighted)
    assert not np.allclose(
        unit_weighted, area_weighted
    ), f"{unit_weighted=}, {area_weighted=}"

    verts = torch.tensor(
        np.stack([a, b, c, d], axis=0),
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)

    normals = compute_vertex_normals(verts=verts, faces=faces, weights="unit")

    expected_unit = torch.tensor(unit_weighted, dtype=torch.float32)
    expected_area = torch.tensor(area_weighted, dtype=torch.float32)
    assert_close(normals[0], expected_unit)
    assert not torch.allclose(
        normals[0], expected_area, atol=1e-4, rtol=0.0
    ), f"{normals[0]=}, {expected_area=}"

    assert_close(normals[2], torch.tensor(n0, dtype=torch.float32))
    assert_close(normals[3], torch.tensor(n1, dtype=torch.float32))


def test_batched_matches_unbatched() -> None:
    """Each batch element's result equals the corresponding single-mesh call.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)
    verts_other = verts * 2.0 + 1.0
    batched = torch.stack([verts, verts_other], dim=0)

    normals_batched = compute_vertex_normals(verts=batched, faces=faces, weights="unit")
    normals_first = compute_vertex_normals(verts=verts, faces=faces, weights="unit")
    normals_second = compute_vertex_normals(
        verts=verts_other, faces=faces, weights="unit"
    )

    assert normals_batched.shape == (2, 4, 3), f"{normals_batched.shape=}"
    assert_close(normals_batched[0], normals_first)
    assert_close(normals_batched[1], normals_second)


def test_area_output_is_unit_length() -> None:
    """Every `weights="area"` per-vertex normal is L2-normalized on a non-degenerate mesh.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)

    normals = compute_vertex_normals(verts=verts, faces=faces, weights="area")

    norms = torch.linalg.norm(normals, dim=-1)
    assert_close(norms, torch.ones_like(norms))


def test_area_weighting_not_unit_weighting() -> None:
    """Shared-edge tent verifies `weights="area"` applies area, not unit, weighting.

    Two triangles share edge A-B with clearly different areas (a non-coplanar
    tent). The shared vertex A's normal must equal
    `normalize(area0 * n0 + area1 * n1)` of the two UN-normalized (area-weighted)
    face normals and must differ from the unit-weighted `normalize(n0 + n1)`,
    since the areas differ. Each apex-only vertex carries the unit normal of its
    single incident face.

    Args:
        None.

    Returns:
        None.
    """

    a = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    c = np.array([0.0, 2.0, 0.0], dtype=np.float64)
    d = np.array([0.0, -1.0, 1.0], dtype=np.float64)

    n0 = _face_unit_normal(v0=a, v1=b, v2=c)
    n1 = _face_unit_normal(v0=a, v1=b, v2=d)
    raw0 = np.cross(a - b, b - c)
    raw1 = np.cross(a - b, b - d)
    area0 = 0.5 * np.linalg.norm(raw0)
    area1 = 0.5 * np.linalg.norm(raw1)
    assert not np.isclose(area0, area1), f"{area0=}, {area1=}"

    unit_weighted = n0 + n1
    unit_weighted = unit_weighted / np.linalg.norm(unit_weighted)
    area_weighted = area0 * n0 + area1 * n1
    area_weighted = area_weighted / np.linalg.norm(area_weighted)
    assert not np.allclose(
        unit_weighted, area_weighted
    ), f"{unit_weighted=}, {area_weighted=}"

    verts = torch.tensor(
        np.stack([a, b, c, d], axis=0),
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 1, 3]], dtype=torch.int64)

    normals = compute_vertex_normals(verts=verts, faces=faces, weights="area")

    expected_unit = torch.tensor(unit_weighted, dtype=torch.float32)
    expected_area = torch.tensor(area_weighted, dtype=torch.float32)
    assert_close(normals[0], expected_area)
    assert not torch.allclose(
        normals[0], expected_unit, atol=1e-4, rtol=0.0
    ), f"{normals[0]=}, {expected_unit=}"

    assert_close(normals[2], torch.tensor(n0, dtype=torch.float32))
    assert_close(normals[3], torch.tensor(n1, dtype=torch.float32))


def test_unrecognized_weights_trips_dispatch_assert() -> None:
    """An unrecognized `weights` value trips the dispatch fall-through assert.

    Args:
        None.

    Returns:
        None.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)

    with pytest.raises(AssertionError):
        compute_vertex_normals(verts=verts, faces=faces, weights="bogus")
