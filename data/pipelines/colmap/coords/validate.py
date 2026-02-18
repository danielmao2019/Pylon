"""Generic camera-center alignment validation for coordinate normalization."""

from typing import List

import numpy as np

CAMERA_CENTER_DIAGONAL_REL_TOL = 1.0e-03


def validate_camera_center_alignment(
    source_camera_names: List[str],
    source_camera_centers: np.ndarray,
    target_camera_names: List[str],
    target_camera_centers: np.ndarray,
    diagonal_rel_tol: float = CAMERA_CENTER_DIAGONAL_REL_TOL,
) -> None:
    """Validate camera-name order and camera-center alignment.

    Compared items:
    - Ordered `source_camera_names` vs ordered `target_camera_names`.
    - Pointwise camera-center pairs from `source_camera_centers` and
      `target_camera_centers` in that order.
    Pass criteria:
    - Camera-name lists are exactly equal (same names and order).
    - max pairwise center error / min(source_bbox_diagonal, target_bbox_diagonal)
      <= `diagonal_rel_tol`.
    Failure mode:
    - Raises `AssertionError` with a structured error message.
    """
    # Input validations
    assert isinstance(source_camera_names, list), f"{type(source_camera_names)=}"
    assert isinstance(
        source_camera_centers, np.ndarray
    ), f"{type(source_camera_centers)=}"
    assert isinstance(target_camera_names, list), f"{type(target_camera_names)=}"
    assert isinstance(
        target_camera_centers, np.ndarray
    ), f"{type(target_camera_centers)=}"
    assert isinstance(diagonal_rel_tol, float), f"{type(diagonal_rel_tol)=}"
    assert diagonal_rel_tol > 0.0, f"{diagonal_rel_tol=}"
    assert (
        source_camera_centers.ndim == 2 and source_camera_centers.shape[1] == 3
    ), f"{source_camera_centers.shape=}"
    assert (
        target_camera_centers.ndim == 2 and target_camera_centers.shape[1] == 3
    ), f"{target_camera_centers.shape=}"

    first_mismatch = _first_name_mismatch(
        source_names=source_camera_names,
        target_names=target_camera_names,
    )
    assert source_camera_names == target_camera_names, (
        "camera_name_order_mismatch "
        f"source_count={len(source_camera_names)} "
        f"target_count={len(target_camera_names)} "
        f"{first_mismatch}"
    )
    assert source_camera_centers.shape == target_camera_centers.shape, (
        "camera_center_shape_mismatch "
        f"source_shape={source_camera_centers.shape} "
        f"target_shape={target_camera_centers.shape}"
    )

    pairwise_errors = np.linalg.norm(
        source_camera_centers - target_camera_centers,
        axis=1,
    )
    max_center_error = float(np.max(pairwise_errors))

    source_diagonal = compute_centers_aabb_diagonal(centers=source_camera_centers)
    target_diagonal = compute_centers_aabb_diagonal(centers=target_camera_centers)
    reference_diagonal = min(source_diagonal, target_diagonal)
    assert reference_diagonal > 0.0, f"{reference_diagonal=}"

    relative_error = max_center_error / reference_diagonal
    assert relative_error <= diagonal_rel_tol, (
        "camera_center_alignment_failed "
        f"max_center_error={max_center_error} "
        f"relative_error={relative_error} "
        f"tol={diagonal_rel_tol} "
        f"source_diagonal={source_diagonal} "
        f"target_diagonal={target_diagonal}"
    )


def compute_centers_aabb_diagonal(centers: np.ndarray) -> float:
    # Input validations
    assert isinstance(centers, np.ndarray), f"{type(centers)=}"
    assert centers.ndim == 2 and centers.shape[1] == 3, f"{centers.shape=}"
    assert centers.shape[0] > 0, f"{centers.shape=}"

    mins = np.min(centers, axis=0)
    maxs = np.max(centers, axis=0)
    diagonal = float(np.linalg.norm(maxs - mins))
    assert diagonal > 0.0, f"{diagonal=}"
    return diagonal


def _first_name_mismatch(
    source_names: List[str],
    target_names: List[str],
) -> str:
    # Input validations
    assert isinstance(source_names, list), f"{type(source_names)=}"
    assert isinstance(target_names, list), f"{type(target_names)=}"

    overlap = min(len(source_names), len(target_names))
    for idx in range(overlap):
        if source_names[idx] != target_names[idx]:
            return (
                f"first_mismatch_index={idx} "
                f"source_name={source_names[idx]} "
                f"target_name={target_names[idx]}"
            )
    if len(source_names) != len(target_names):
        return f"prefix_match_only overlap={overlap}"
    return "lists_equal"
