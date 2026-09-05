"""Unit tests for the UV-origin convention transform."""

import torch

from data.structures.three_d.mesh.texture.conventions import transform_uv_convention


def test_identity_when_conventions_match() -> None:
    """Return the UV table unchanged when the conventions are equal.

    Args:
        None.

    Returns:
        None.
    """

    verts_uvs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]],
        dtype=torch.float32,
    )

    transformed = transform_uv_convention(
        verts_uvs=verts_uvs,
        source_uv_convention="obj",
        target_uv_convention="obj",
    )

    assert transformed is verts_uvs, f"{transformed=} {verts_uvs=}"


def test_flips_v_axis_when_conventions_differ() -> None:
    """Flip the V axis (v -> 1 - v) when the conventions differ.

    Args:
        None.

    Returns:
        None.
    """

    verts_uvs = torch.tensor(
        [[0.0, 0.0], [1.0, 0.25], [0.5, 1.0]],
        dtype=torch.float32,
    )

    transformed = transform_uv_convention(
        verts_uvs=verts_uvs,
        source_uv_convention="obj",
        target_uv_convention="top_left",
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
        transformed[:, 0], verts_uvs[:, 0]
    ), f"{transformed[:, 0]=} {verts_uvs[:, 0]=}"
