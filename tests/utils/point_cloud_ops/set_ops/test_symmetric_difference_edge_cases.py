import torch
import pytest
from data.structures.three_d.point_cloud.ops.set_ops.symmetric_difference import (
    pc_symmetric_difference,
    _normalize_points,
)


def test_normalize_points_invalid_shape():
    """Test _normalize_points with invalid input shapes."""
    # Test 1D input
    with pytest.raises(ValueError, match="Points must have 2 or 3 dimensions"):
        points_1d = torch.tensor([1.0, 2.0, 3.0])
        _normalize_points(points_1d)

    # Test 4D input
    with pytest.raises(ValueError, match="Points must have 2 or 3 dimensions"):
        points_4d = torch.randn(1, 2, 3, 4)
        _normalize_points(points_4d)

    # Test batch size > 1
    with pytest.raises(AssertionError, match="Batch size must be 1"):
        points_batch = torch.randn(2, 10, 3)
        _normalize_points(points_batch)

    # Test wrong coordinate dimension
    with pytest.raises(AssertionError, match="Points must have 3 coordinates"):
        points_wrong_coords = torch.randn(10, 2)
        _normalize_points(points_wrong_coords)


def test_pc_symmetric_difference_identical_points():
    """Test symmetric difference with identical point clouds."""
    src_pc = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Identical target points
    tgt_pc = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # Identical to src[0]
            [1.0, 1.0, 1.0],  # Identical to src[1]
        ],
        dtype=torch.float32,
    )

    radius = 0.1

    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    expected_empty = torch.tensor([], dtype=torch.long)

    # Should be no symmetric difference for identical points within radius
    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)


def test_pc_symmetric_difference_single_point():
    """Test symmetric difference with single-point clouds."""
    src_pc = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    tgt_pc = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)

    # Close points - should not be in symmetric difference
    radius = 0.5
    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    expected_empty = torch.tensor([], dtype=torch.long)
    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)

    # Far points - should be in symmetric difference
    tgt_pc_far = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc_far, radius)
    expected_single = torch.tensor([0], dtype=torch.long)
    assert torch.equal(src_indices, expected_single)
    assert torch.equal(tgt_indices, expected_single)


def test_pc_symmetric_difference_very_small_radius():
    """Test symmetric difference with very small radius."""
    src_pc = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    # Slightly offset target points
    tgt_pc = torch.tensor(
        [
            [0.0001, 0.0, 0.0],  # Very close to src[0]
            [1.0001, 0.0, 0.0],  # Very close to src[1]
        ],
        dtype=torch.float32,
    )

    radius = 0.001  # Very small radius - should still match close points

    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    expected_empty = torch.tensor([], dtype=torch.long)

    # Should be no symmetric difference for very close points
    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)


def test_pc_symmetric_difference_very_large_radius():
    """Test symmetric difference with very large radius."""
    src_pc = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=torch.float32,
    )

    tgt_pc = torch.tensor(
        [
            [50.0, 50.0, 50.0],
            [200.0, 200.0, 200.0],
        ],
        dtype=torch.float32,
    )

    radius = 1000.0  # Very large radius - all points should match

    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    expected_empty = torch.tensor([], dtype=torch.long)

    # Should be no symmetric difference with large radius
    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)
