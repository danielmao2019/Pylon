"""Unit tests for the mesh `apply_transform` op."""

import pytest
import torch
from torch.testing import assert_close

from data.structures.three_d.mesh.mesh import Mesh
from data.structures.three_d.mesh.texture.mesh_texture_vertex_color import (
    MeshTextureVertexColor,
)
from models.three_d.meshes.ops.apply_transform import apply_transform


def _make_mesh() -> Mesh:
    """Build one small textured mesh fixture.

    Args:
        None.

    Returns:
        A `Mesh` with float32 `[V, 3]` verts, int64 `[F, 3]` faces, and a
        per-vertex color texture.
    """

    verts = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=torch.int64,
    )
    texture = MeshTextureVertexColor(
        vertex_color=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )
    )
    return Mesh(verts=verts, faces=faces, texture=texture)


def _make_transform() -> torch.Tensor:
    """Build one non-trivial 4x4 affine transform.

    Args:
        None.

    Returns:
        A float32 `[4, 4]` transform combining rotation, scaling, and
        translation.
    """

    return torch.tensor(
        [
            [0.0, -2.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, -2.0],
            [0.0, 0.0, 0.5, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_verts_match_reference_matmul() -> None:
    """Transformed verts equal a direct homogeneous matmul of `mesh.verts`."""

    mesh = _make_mesh()
    transform = _make_transform()

    ones_column = torch.ones(
        (mesh.verts.shape[0], 1), dtype=mesh.verts.dtype, device=mesh.verts.device
    )
    homogeneous_verts = torch.cat([mesh.verts, ones_column], dim=1)
    reference = torch.matmul(homogeneous_verts, transform.T)[:, :3]

    transformed = apply_transform(mesh=mesh, transform=transform)

    assert_close(transformed.verts, reference)


def test_faces_and_texture_preserved() -> None:
    """The returned `Mesh` keeps the original faces and texture unchanged."""

    mesh = _make_mesh()
    transform = _make_transform()

    transformed = apply_transform(mesh=mesh, transform=transform)

    assert_close(transformed.faces, mesh.faces)
    assert transformed.texture is mesh.texture


def test_rejects_non_4x4_transform() -> None:
    """A transform that is not a `[4, 4]` matrix raises an assertion."""

    mesh = _make_mesh()
    bad_transform = torch.eye(3, dtype=torch.float32)

    with pytest.raises(AssertionError):
        apply_transform(mesh=mesh, transform=bad_transform)
