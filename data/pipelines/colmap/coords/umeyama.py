"""Umeyama alignment utilities for coordinate normalization."""

from typing import Tuple

import numpy as np


def compute_umeyama_alignment(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute similarity transform aligning source points to target points.

    Input shapes are `(3, N)` where columns are points.
    Returns `(scale, rotation, translation)` such that:
    `target ~= scale * rotation @ source + translation`.
    """
    # Input validations
    assert isinstance(source_points, np.ndarray), f"{type(source_points)=}"
    assert isinstance(target_points, np.ndarray), f"{type(target_points)=}"
    assert (
        source_points.shape == target_points.shape
    ), f"{source_points.shape=} {target_points.shape=}"
    assert (
        source_points.ndim == 2 and source_points.shape[0] == 3
    ), f"{source_points.shape=}"
    assert source_points.dtype == np.float32, f"{source_points.dtype=}"
    assert target_points.dtype == np.float32, f"{target_points.dtype=}"

    dimension, num_points = source_points.shape
    source_mean = source_points.mean(axis=1, keepdims=True)
    target_mean = target_points.mean(axis=1, keepdims=True)
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean

    source_variance = np.sum(source_centered**2) / num_points
    covariance = (target_centered @ source_centered.T) / num_points
    U, singular_values, VH = np.linalg.svd(covariance)

    sign_matrix = np.eye(dimension, dtype=source_points.dtype)
    if np.linalg.det(U) * np.linalg.det(VH) < 0:
        sign_matrix[-1, -1] = -1

    scale = np.trace(np.diag(singular_values) @ sign_matrix) / source_variance
    rotation = U @ sign_matrix @ VH
    translation = (target_mean - scale * rotation @ source_mean).flatten()

    assert isinstance(scale, np.floating) or isinstance(scale, float), f"{type(scale)=}"
    assert isinstance(rotation, np.ndarray), f"{type(rotation)=}"
    assert isinstance(translation, np.ndarray), f"{type(translation)=}"
    assert rotation.dtype == np.float32, f"{rotation.dtype=}"
    assert translation.dtype == np.float32, f"{translation.dtype=}"
    return float(scale), rotation, translation
