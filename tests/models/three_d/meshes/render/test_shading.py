"""Unit tests for the spherical-harmonic shading op."""

import pytest
import torch

from models.three_d.meshes.render.shading import compute_sh_shading


def test_band_count_selects_the_spherical_harmonic_order() -> None:
    """Perfect-square band counts evaluate at the spherical-harmonic order each count implies.

    Args:
        None.

    Returns:
        None.
    """

    normals = torch.nn.functional.normalize(
        torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 2.0, -3.0],
            ],
            dtype=torch.float32,
        ),
        dim=-1,
        p=2,
    )
    # The sampled perfect-square band counts include one higher than 9 bands.
    for num_bands in (1, 4, 9, 16):
        sh_coefficients = torch.arange(num_bands * 3, dtype=torch.float32) / float(
            num_bands * 3
        )
        shading = compute_sh_shading(normals=normals, sh_coefficients=sh_coefficients)
        assert shading.shape == (normals.shape[0], 3), (
            "Expected one RGB shading triple per input normal. "
            f"{shading.shape=}, {normals.shape=}, {num_bands=}"
        )


def test_non_square_band_count_is_rejected() -> None:
    """A coefficient set whose band count is not a perfect square names no spherical-harmonic order, so it fails the assertion rather than evaluating.

    Args:
        None.

    Returns:
        None.
    """

    normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    # 6 = 2 bands x 3 channels; 2 is not a perfect square, so no order is named.
    sh_coefficients = torch.zeros(6, dtype=torch.float32)
    with pytest.raises(AssertionError):
        compute_sh_shading(normals=normals, sh_coefficients=sh_coefficients)


def test_higher_order_coefficients_affect_shading() -> None:
    """Coefficient bands above degree 2 participate in the shading result.

    Args:
        None.

    Returns:
        None.
    """

    normals = torch.nn.functional.normalize(
        torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32),
        dim=-1,
        p=2,
    )
    sh_coefficients = torch.zeros(16 * 3, dtype=torch.float32)
    sh_coefficients[9:16] = 1.0
    shading = compute_sh_shading(normals=normals, sh_coefficients=sh_coefficients)
    assert not torch.allclose(shading, torch.zeros_like(shading)), (
        "Expected degree-3 coefficients to contribute to the shading. "
        f"{shading=}, {normals=}, {sh_coefficients=}"
    )


def test_shading_varies_with_the_normal_direction() -> None:
    """Two normals facing differently under the same non-constant coefficients receive different shading, so the basis is really evaluated over the normal.

    Args:
        None.

    Returns:
        None.
    """

    normals = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    # Non-constant lighting: every channel carries the same nonzero degree-1 band on top of a zero DC band.
    sh_coefficients = torch.tensor(
        [0.0, 0.5, -0.25, 0.75] * 3,
        dtype=torch.float32,
    )
    shading = compute_sh_shading(normals=normals, sh_coefficients=sh_coefficients)
    assert not torch.allclose(shading[0], shading[1]), (
        "Expected two differently facing normals to receive different shading. "
        f"{shading=}, {normals=}, {sh_coefficients=}"
    )
