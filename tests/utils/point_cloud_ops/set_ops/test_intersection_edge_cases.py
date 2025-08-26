import torch
import pytest
from utils.point_cloud_ops.set_ops.intersection import (
    pc_intersection,
    compute_pc_iou,
    get_nearest_neighbor_distances,
    compute_registration_overlap,
    _tensor_intersection,
    _kdtree_intersection
)


def test_tensor_intersection_identical_points():
    """Test tensor intersection with identical point clouds."""
    identical_pc = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    
    radius = 0.1
    
    # Identical point clouds should have full intersection
    src_indices, tgt_indices = _tensor_intersection(identical_pc, identical_pc, radius)
    expected_indices = torch.tensor([0, 1], dtype=torch.long)
    assert torch.equal(torch.sort(src_indices)[0], expected_indices)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_indices)


def test_kdtree_intersection_identical_points():
    """Test KDTree intersection with identical point clouds."""
    identical_pc = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    
    radius = 0.1
    
    # Identical point clouds should have full intersection
    src_indices, tgt_indices = _kdtree_intersection(identical_pc, identical_pc, radius)
    expected_indices = torch.tensor([0, 1], dtype=torch.long)
    assert torch.equal(torch.sort(src_indices)[0], expected_indices)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_indices)


def test_pc_intersection_single_point():
    """Test intersection with single-point clouds."""
    src_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    tgt_points = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)
    
    # Close points - should intersect
    radius = 0.5
    src_indices, tgt_indices = pc_intersection(src_points, tgt_points, radius)
    expected_single = torch.tensor([0], dtype=torch.long)
    assert torch.equal(src_indices, expected_single)
    assert torch.equal(tgt_indices, expected_single)
    
    # Far points - should not intersect
    tgt_points_far = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    src_indices, tgt_indices = pc_intersection(src_points, tgt_points_far, radius)
    expected_empty = torch.tensor([], dtype=torch.long)
    assert torch.equal(src_indices, expected_empty)
    assert torch.equal(tgt_indices, expected_empty)


def test_compute_pc_iou_perfect_overlap():
    """Test IoU computation with perfect overlap (identical points)."""
    identical_pc = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    
    radius = 0.1
    iou = compute_pc_iou(identical_pc, identical_pc, radius)
    assert iou == 1.0  # All points overlap


def test_compute_pc_iou_no_overlap():
    """Test IoU computation with no overlap (far apart points)."""
    pc1 = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    pc2 = torch.tensor([
        [10.0, 0.0, 0.0],
        [11.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    radius = 0.1
    iou = compute_pc_iou(pc1, pc2, radius)
    assert iou == 0.0  # No points overlap


def test_get_nearest_neighbor_distances_single_query_single_support():
    """Test nearest neighbor distance computation with single query and support point."""
    query_single = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    support_single = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    
    distances = get_nearest_neighbor_distances(query_single, support_single)
    assert distances.shape == (1,)
    assert abs(distances[0].item() - 1.0) < 1e-6


def test_get_nearest_neighbor_distances_multiple_query_single_support():
    """Test nearest neighbor distance computation with multiple queries and single support point."""
    query_multiple = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=torch.float32)
    support_single = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    
    distances = get_nearest_neighbor_distances(query_multiple, support_single)
    assert distances.shape == (2,)
    assert abs(distances[0].item() - 1.0) < 1e-6  # Distance to [1,0,0]
    assert abs(distances[1].item() - 1.0) < 1e-6  # Distance to [1,0,0]


def test_compute_registration_overlap_close_single_points():
    """Test registration overlap computation with close single points."""
    single_ref = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    single_src = torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32)
    
    positive_radius = 0.1
    overlap = compute_registration_overlap(single_ref, single_src, None, positive_radius)
    assert overlap == 1.0


def test_compute_registration_overlap_far_single_points():
    """Test registration overlap computation with far single points."""
    single_ref = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    far_src = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
    
    positive_radius = 0.1
    overlap = compute_registration_overlap(single_ref, far_src, None, positive_radius)
    assert overlap == 0.0


def test_compute_registration_overlap_identity_transform():
    """Test registration overlap computation with identity transformation."""
    ref_points = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    src_points = torch.tensor([[0.05, 0.0, 0.0]], dtype=torch.float32)
    
    positive_radius = 0.1
    identity_transform = torch.eye(4, dtype=torch.float32)
    overlap = compute_registration_overlap(ref_points, src_points, identity_transform, positive_radius)
    assert overlap == 1.0  # Single ref point has close src neighbor


def test_intersection_very_small_radius():
    """Test intersection functions with very small radius."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ], dtype=torch.float32)
    
    # Identical target points
    tgt_points = torch.tensor([
        [0.0, 0.0, 0.0],  # Identical to src[0]
        [1.0, 0.0, 0.0],  # Identical to src[1]
    ], dtype=torch.float32)
    
    radius = 0.001  # Very small radius - only identical points intersect
    
    src_indices, tgt_indices = _tensor_intersection(src_points, tgt_points, radius)
    expected_both = torch.tensor([0, 1], dtype=torch.long)
    
    # Should find intersection for identical points
    assert torch.equal(torch.sort(src_indices)[0], expected_both)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_both)


def test_intersection_very_large_radius():
    """Test intersection functions with very large radius."""
    src_points = torch.tensor([
        [0.0, 0.0, 0.0],
        [100.0, 100.0, 100.0],
    ], dtype=torch.float32)
    
    tgt_points = torch.tensor([
        [50.0, 50.0, 50.0],
        [200.0, 200.0, 200.0],
    ], dtype=torch.float32)
    
    radius = 1000.0  # Very large radius - all points should intersect
    
    src_indices, tgt_indices = _tensor_intersection(src_points, tgt_points, radius)
    expected_both = torch.tensor([0, 1], dtype=torch.long)
    
    # Should find all points in intersection with large radius
    assert torch.equal(torch.sort(src_indices)[0], expected_both)
    assert torch.equal(torch.sort(tgt_indices)[0], expected_both)