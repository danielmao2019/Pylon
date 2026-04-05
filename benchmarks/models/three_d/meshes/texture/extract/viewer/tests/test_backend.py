"""Focused backend tests for the texture-extraction benchmark viewer."""

import numpy as np
import torch

from benchmarks.models.three_d.meshes.texture.extract.viewer.backend.benchmark_backend import (
    _apply_invisible_texel_noise,
)


def test_apply_invisible_texel_noise_uses_seed_zero_rgb_noise() -> None:
    """Invisible occupied texels should receive deterministic seed-zero RGB noise.

    Args:
        None.

    Returns:
        None.
    """

    uv_texture_map = torch.full((1, 2, 2, 3), fill_value=0.7, dtype=torch.float32)
    uv_valid_mask = torch.tensor(
        [[[[1.0], [0.0]], [[0.0], [0.0]]]],
        dtype=torch.float32,
    )
    uv_occupancy_mask = torch.tensor(
        [[[[1.0], [1.0]], [[0.0], [1.0]]]],
        dtype=torch.float32,
    )

    output = _apply_invisible_texel_noise(
        uv_texture_map=uv_texture_map,
        uv_valid_mask=uv_valid_mask,
        uv_occupancy_mask=uv_occupancy_mask,
    )

    expected_noise = np.random.default_rng(seed=0).random(
        size=(1, 2, 2, 3),
        dtype=np.float32,
    )
    np.testing.assert_allclose(
        output[0, 0, 0].numpy(),
        np.array([0.7, 0.7, 0.7], dtype=np.float32),
    )
    np.testing.assert_allclose(
        output[0, 0, 1].numpy(),
        expected_noise[0, 0, 1],
    )
    np.testing.assert_allclose(
        output[0, 1, 1].numpy(),
        expected_noise[0, 1, 1],
    )
    np.testing.assert_allclose(
        output[0, 1, 0].numpy(),
        np.zeros(3, dtype=np.float32),
    )
