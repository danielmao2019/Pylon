import torch
import pytest
from data.structures.three_d.point_cloud.ops.set_ops.intersection import (
    pc_intersection,
    compute_pc_iou,
    get_nearest_neighbor_distances,
    compute_registration_overlap,
    _calculate_chunk_factor,
    _tensor_intersection,
    _tensor_intersection_recursive,
    _kdtree_intersection,
)


def test_tensor_intersection_basic():
    """Test basic tensor intersection functionality."""
    # Create two point clouds with some overlap
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Close to tgt[0]
            [1.0, 0.0, 0.0],  # Far from all tgt points
            [2.0, 0.0, 0.0],  # Close to tgt[1]
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to src[0]
            [2.1, 0.0, 0.0],  # Close to src[2]
            [5.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    src_indices, tgt_indices = _tensor_intersection(src_points, tgt_points, radius)

    # Should find src[0,2] and tgt[0,1] in intersection
    expected_src = torch.tensor([0, 2], dtype=torch.long)
    expected_tgt = torch.tensor([0, 1], dtype=torch.long)

    assert torch.equal(torch.sort(src_indices)[0], expected_src)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_tgt)


def test_tensor_intersection_no_overlap():
    """Test tensor intersection when no points overlap."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [10.0, 0.0, 0.0],  # Far from all src points
            [20.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    src_indices, tgt_indices = _tensor_intersection(src_points, tgt_points, radius)

    # No points should overlap
    expected_empty = torch.tensor([], dtype=torch.long)

    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)


def test_kdtree_intersection_basic():
    """Test basic KDTree intersection functionality."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Close to tgt[0]
            [1.0, 0.0, 0.0],  # Far from all tgt points
            [2.0, 0.0, 0.0],  # Close to tgt[1]
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to src[0]
            [2.1, 0.0, 0.0],  # Close to src[2]
            [5.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    src_indices, tgt_indices = _kdtree_intersection(src_points, tgt_points, radius)

    # Should find src[0,2] and tgt[0,1] in intersection
    expected_src = torch.tensor([0, 2], dtype=torch.long)
    expected_tgt = torch.tensor([0, 1], dtype=torch.long)

    assert torch.equal(torch.sort(src_indices)[0], expected_src)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_tgt)


def test_pc_intersection_basic():
    """Test main pc_intersection function (uses _tensor_intersection_recursive)."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Close to tgt[0]
            [1.0, 0.0, 0.0],  # Far from all tgt points
            [2.0, 0.0, 0.0],  # Close to tgt[1]
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to src[0]
            [2.1, 0.0, 0.0],  # Close to src[2]
            [5.0, 0.0, 0.0],  # Far from all src points
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    src_indices, tgt_indices = pc_intersection(src_points, tgt_points, radius)

    # Should find src[0,2] and tgt[0,1] in intersection
    expected_src = torch.tensor([0, 2], dtype=torch.long)
    expected_tgt = torch.tensor([0, 1], dtype=torch.long)

    assert torch.equal(torch.sort(src_indices)[0], expected_src)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_tgt)


def test_compute_pc_iou():
    """Test IoU computation between two point clouds."""
    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Close to tgt[0] - overlapping
            [1.0, 0.0, 0.0],  # Far from all tgt points - not overlapping
        ],
        dtype=torch.float32,
    )

    tgt_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to src[0] - overlapping
            [5.0, 0.0, 0.0],  # Far from all src points - not overlapping
        ],
        dtype=torch.float32,
    )

    radius = 0.5

    iou = compute_pc_iou(src_points, tgt_points, radius)

    # 2 overlapping points out of 4 total points = 0.5
    expected_iou = 2.0 / 4.0
    assert abs(iou - expected_iou) < 1e-6


def test_get_nearest_neighbor_distances():
    """Test nearest neighbor distance computation."""
    query_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    support_points = torch.tensor(
        [
            [0.1, 0.0, 0.0],  # Close to query[0]
            [1.2, 0.0, 0.0],  # Close to query[1]
            [10.0, 0.0, 0.0],  # Far from all query points
        ],
        dtype=torch.float32,
    )

    distances = get_nearest_neighbor_distances(query_points, support_points)

    # Check shape
    assert distances.shape == (3,)

    # Calculate expected distances manually:
    # query[0] [0,0,0] distances to supports: [0.1, 1.2, 10.0] -> nearest is 0.1
    # query[1] [1,0,0] distances to supports: [0.9, 0.2, 9.0] -> nearest is 0.2
    # query[2] [5,0,0] distances to supports: [4.9, 3.8, 5.0] -> nearest is 3.8
    expected_distances = torch.tensor([0.1, 0.2, 3.8], dtype=torch.float32)

    assert torch.allclose(distances, expected_distances, atol=1e-6)


def test_compute_registration_overlap_no_transform():
    """Test registration overlap computation without transformation."""
    ref_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # Far from src points
        ],
        dtype=torch.float32,
    )

    src_points = torch.tensor(
        [
            [0.05, 0.0, 0.0],  # Close to ref[0]
            [1.05, 0.0, 0.0],  # Close to ref[1]
        ],
        dtype=torch.float32,
    )

    positive_radius = 0.1

    overlap = compute_registration_overlap(
        ref_points, src_points, None, positive_radius
    )

    # 2 out of 3 ref points have close src neighbors
    expected_overlap = 2.0 / 3.0
    assert abs(overlap - expected_overlap) < 1e-6


def test_compute_registration_overlap_with_transform():
    """Test registration overlap computation with transformation."""
    ref_points = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    src_points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Will become [1.0, 0.0, 0.0] after transform
            [1.0, 0.0, 0.0],  # Will become [2.0, 0.0, 0.0] after transform
        ],
        dtype=torch.float32,
    )

    # Translation by [1, 0, 0]
    transform = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    positive_radius = 0.1

    overlap = compute_registration_overlap(
        ref_points, src_points, transform, positive_radius
    )

    # After transformation, all ref points should have close src neighbors
    expected_overlap = 1.0
    assert abs(overlap - expected_overlap) < 1e-6


def test_calculate_chunk_factor_cpu():
    """Test chunk factor calculation for CPU tensors."""
    src_points = torch.randn(100, 3, dtype=torch.float32)
    tgt_points = torch.randn(100, 3, dtype=torch.float32)

    chunk_factor = _calculate_chunk_factor(src_points, tgt_points)

    # CPU should always return chunk_factor = 1
    assert chunk_factor == 1
