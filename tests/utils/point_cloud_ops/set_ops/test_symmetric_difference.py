import torch
import pytest
from utils.point_cloud_ops.set_ops.symmetric_difference import pc_symmetric_difference, _normalize_points


def test_normalize_points_unbatched():
    """Test _normalize_points with unbatched input [N, 3]."""
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    
    normalized = _normalize_points(points)
    
    assert normalized.shape == (2, 3)
    assert torch.equal(normalized, points)


def test_normalize_points_batched():
    """Test _normalize_points with batched input [1, N, 3]."""
    points = torch.tensor([[
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]], dtype=torch.float32)
    
    normalized = _normalize_points(points)
    
    assert normalized.shape == (2, 3)
    expected = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    assert torch.equal(normalized, expected)


def test_pc_symmetric_difference_basic():
    """Test basic symmetric difference functionality."""
    # Create two point clouds with some overlap
    src_pc = torch.tensor([
        [0.0, 0.0, 0.0],  # Close to tgt[0], not in difference
        [1.0, 0.0, 0.0],  # Far from all tgt points, in difference
        [2.0, 0.0, 0.0],  # Close to tgt[1], not in difference
    ], dtype=torch.float32)
    
    tgt_pc = torch.tensor([
        [0.1, 0.0, 0.0],  # Close to src[0], not in difference
        [2.1, 0.0, 0.0],  # Close to src[2], not in difference
        [5.0, 0.0, 0.0],  # Far from all src points, in difference
    ], dtype=torch.float32)
    
    radius = 0.5
    
    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    
    # Should find src[1] and tgt[2] in symmetric difference
    expected_src = torch.tensor([1], dtype=torch.long)
    expected_tgt = torch.tensor([2], dtype=torch.long)
    
    assert torch.equal(src_indices, expected_src)
    assert torch.equal(tgt_indices, expected_tgt)


def test_pc_symmetric_difference_no_overlap():
    """Test symmetric difference when no points overlap."""
    src_pc = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_pc = torch.tensor([
        [10.0, 0.0, 0.0],  # Far from all src points
        [20.0, 0.0, 0.0],  # Far from all src points
    ], dtype=torch.float32)
    
    radius = 0.5
    
    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    
    # All points should be in symmetric difference
    expected_src = torch.tensor([0, 1], dtype=torch.long)
    expected_tgt = torch.tensor([0, 1], dtype=torch.long)
    
    assert torch.equal(src_indices, expected_src)
    assert torch.equal(tgt_indices, expected_tgt)


def test_pc_symmetric_difference_full_overlap():
    """Test symmetric difference when all points overlap."""
    src_pc = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    tgt_pc = torch.tensor([
        [0.1, 0.0, 0.0],  # Close to src[0]
        [1.1, 0.0, 0.0],  # Close to src[1]
    ], dtype=torch.float32)
    
    radius = 0.5
    
    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    
    # No points should be in symmetric difference
    expected_empty = torch.tensor([], dtype=torch.long)
    
    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)


def test_pc_symmetric_difference_batched_input():
    """Test symmetric difference with batched input [1, N, 3]."""
    src_pc = torch.tensor([[
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],  # Far from tgt points
    ]], dtype=torch.float32)
    
    tgt_pc = torch.tensor([[
        [0.1, 0.0, 0.0],  # Close to src[0]
        [10.0, 0.0, 0.0],  # Far from src points
    ]], dtype=torch.float32)
    
    radius = 0.5
    
    src_indices, tgt_indices = pc_symmetric_difference(src_pc, tgt_pc, radius)
    
    # Should find src[1] and tgt[1] in symmetric difference
    expected_src = torch.tensor([1], dtype=torch.long)
    expected_tgt = torch.tensor([1], dtype=torch.long)
    
    assert torch.equal(src_indices, expected_src)
    assert torch.equal(tgt_indices, expected_tgt)