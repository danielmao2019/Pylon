"""Shared helpers for recovering 3DGS scene normalization."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def compute_scene_normalization(
    model_dir: Union[str, Path],
) -> Tuple[np.ndarray, float]:
    """Reconstruct the translate/scale used by the original GraphDECO pipelines.

    Args:
        model_dir: Directory containing the trained checkpoint (with ``cfg_args``).

    Returns:
        A tuple ``(center, scale)`` where ``center`` is the world-space centre
        used for normalization (as a float32 numpy array of shape ``[3]``) and
        ``scale`` is the scalar radius factor (float).

    Raises:
        FileNotFoundError: If the loader cannot locate the metadata required to
            reconstruct the normalization parameters.
        ValueError: If the metadata exists but does not contain the expected
            fields.
    """

    model_path = Path(model_dir)
    cfg_path = model_path / "cfg_args"
    assert (
        cfg_path.is_file()
    ), f"Expected cfg_args in '{model_path}' to recover normalization"

    cfg_text = cfg_path.read_text()
    match = re.search(r"source_path='([^']+)'", cfg_text)
    assert (
        match is not None
    ), "cfg_args does not contain a 'source_path' entry; cannot rebuild normalization"

    source_path = Path(match.group(1))
    transforms_path = source_path / "transforms.json"
    assert (
        transforms_path.is_file()
    ), f"Expected transforms.json at '{transforms_path}' to compute normalization"

    with open(transforms_path, "r", encoding="utf-8") as f:
        transforms = json.load(f)

    frames = transforms.get("frames")
    assert frames, f"transforms.json at '{transforms_path}' does not contain any frames"

    centers = []
    for frame in frames:
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        assert matrix.shape == (
            4,
            4,
        ), "transform_matrix entries must be 4x4 to recover camera centres"
        centers.append(matrix[:3, 3])

    centers_np = np.stack(centers, axis=0)
    center = centers_np.mean(axis=0)
    distances = np.linalg.norm(centers_np - center[None, :], axis=1)
    max_dist = float(distances.max())
    assert (
        np.isfinite(max_dist) and max_dist > 0.0
    ), "Could not compute a valid normalization radius from camera poses"

    radius = 1.1 * max_dist
    return center.astype(np.float32), radius
