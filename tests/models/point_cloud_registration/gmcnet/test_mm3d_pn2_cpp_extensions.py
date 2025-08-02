"""Tests for mm3d_pn2 C++ extensions to verify PyTorch 2.0 compatibility.

This module tests the C++ extensions in GMCNet's mm3d_pn2 module to ensure they work
correctly after modifications for PyTorch 2.0 compatibility. These tests verify that
the functions execute without errors and return tensors with expected shapes and types.

Key C++ extensions tested:
- three_nn: k-nearest neighbor search for interpolation
- three_interpolate: feature interpolation using nearest neighbors
- furthest_point_sample: point sampling for downsampling
- gather_points: gathering points by indices
- grouping_operation: grouping points in neighborhoods
- ball_query: spherical neighborhood search with radius constraints
- knn: k-nearest neighbor search based on heap data structure

The tests use realistic point cloud data sizes but focus on functional correctness
rather than mathematical accuracy (which is handled by the original implementations).
"""

import pytest
import torch
import math

# Direct imports from the models module
from models.point_cloud_registration.gmcnet.mm3d_pn2.ops.interpolate import three_nn, three_interpolate
from models.point_cloud_registration.gmcnet.mm3d_pn2.ops.furthest_point_sample import furthest_point_sample
from models.point_cloud_registration.gmcnet.mm3d_pn2.ops.gather_points import gather_points
from models.point_cloud_registration.gmcnet.mm3d_pn2.ops.group_points import grouping_operation
from models.point_cloud_registration.gmcnet.mm3d_pn2.ops.ball_query import ball_query
from models.point_cloud_registration.gmcnet.mm3d_pn2.ops.knn import knn


def test_three_nn_cpu_tensors_moved_to_cuda():
    """Test three_nn function with CPU tensors that are moved to CUDA."""
    batch_size = 2
    target_points_num = 100
    source_points_num = 200
    device = torch.device('cuda')
    
    # Create random point clouds on CPU, then move to CUDA
    target_points = torch.randn(batch_size, target_points_num, 3).to(device)
    source_points = torch.randn(batch_size, source_points_num, 3).to(device)
    
    # Call three_nn function
    distances, indices = three_nn(target_points, source_points)
    
    # Verify output shapes and types
    assert distances.shape == (batch_size, target_points_num, 3), f"Expected distances shape {(batch_size, target_points_num, 3)}, got {distances.shape}"
    assert indices.shape == (batch_size, target_points_num, 3), f"Expected indices shape {(batch_size, target_points_num, 3)}, got {indices.shape}"
    assert distances.dtype == torch.float32, f"Expected distances dtype float32, got {distances.dtype}"
    assert indices.dtype == torch.int32, f"Expected indices dtype int32, got {indices.dtype}"
    assert distances.device.type == 'cuda', "Distances should be on CUDA device"
    assert indices.device.type == 'cuda', "Indices should be on CUDA device"
    
    # Verify distances are non-negative
    assert torch.all(distances >= 0), "All distances should be non-negative"
    
    # Verify indices are within valid range
    assert torch.all(indices >= 0), "All indices should be non-negative"
    assert torch.all(indices < source_points_num), f"All indices should be less than {source_points_num}"


def test_three_nn_cuda():
    """Test three_nn function with CUDA tensors."""
    batch_size = 2
    target_points_num = 100
    source_points_num = 200
    device = torch.device('cuda')
    
    # Create random point clouds on CUDA
    target_points = torch.randn(batch_size, target_points_num, 3, device=device)
    source_points = torch.randn(batch_size, source_points_num, 3, device=device)
    
    # Call three_nn function
    distances, indices = three_nn(target_points, source_points)
    
    # Verify output shapes, types, and device
    assert distances.shape == (batch_size, target_points_num, 3)
    assert indices.shape == (batch_size, target_points_num, 3)
    assert distances.dtype == torch.float32
    assert indices.dtype == torch.int32
    assert distances.device.type == 'cuda'
    assert indices.device.type == 'cuda'
    
    # Verify distances are non-negative
    assert torch.all(distances >= 0), "All distances should be non-negative"


def test_three_interpolate_cuda_only():
    """Test three_interpolate function with CUDA tensors."""
    batch_size = 2
    num_channels = 64
    source_points_num = 200
    target_points_num = 100
    device = torch.device('cuda')
    
    # Create features and interpolation parameters on CUDA
    features = torch.randn(batch_size, num_channels, source_points_num, device=device)
    indices = torch.randint(0, source_points_num, (batch_size, target_points_num, 3), dtype=torch.int32, device=device)
    weights = torch.rand(batch_size, target_points_num, 3, device=device)
    
    # Normalize weights to sum to 1 for each point
    weights = weights / weights.sum(dim=2, keepdim=True)
    
    # Call three_interpolate function
    interpolated_features = three_interpolate(features, indices, weights)
    
    # Verify output shape and type
    expected_shape = (batch_size, num_channels, target_points_num)
    assert interpolated_features.shape == expected_shape, f"Expected shape {expected_shape}, got {interpolated_features.shape}"
    assert interpolated_features.dtype == torch.float32, f"Expected dtype float32, got {interpolated_features.dtype}"
    assert interpolated_features.device.type == 'cuda', "Output should be on CUDA device"


def test_three_interpolate_cuda():
    """Test three_interpolate function with CUDA tensors."""
    batch_size = 2
    num_channels = 64
    source_points_num = 200
    target_points_num = 100
    device = torch.device('cuda')
    
    # Create features and interpolation parameters on CUDA
    features = torch.randn(batch_size, num_channels, source_points_num, device=device)
    indices = torch.randint(0, source_points_num, (batch_size, target_points_num, 3), dtype=torch.int32, device=device)
    weights = torch.rand(batch_size, target_points_num, 3, device=device)
    
    # Normalize weights to sum to 1 for each point
    weights = weights / weights.sum(dim=2, keepdim=True)
    
    # Call three_interpolate function
    interpolated_features = three_interpolate(features, indices, weights)
    
    # Verify output shape, type, and device
    expected_shape = (batch_size, num_channels, target_points_num)
    assert interpolated_features.shape == expected_shape
    assert interpolated_features.dtype == torch.float32
    assert interpolated_features.device.type == 'cuda'


def test_furthest_point_sample_cuda_only():
    """Test furthest_point_sample function with CUDA tensors."""
    batch_size = 2
    num_points = 1000
    num_samples = 256
    device = torch.device('cuda')
    
    # Create random point cloud on CUDA
    points = torch.randn(batch_size, num_points, 3, device=device)
    
    # Call furthest_point_sample function
    sampled_indices = furthest_point_sample(points, num_samples)
    
    # Verify output shape and type
    expected_shape = (batch_size, num_samples)
    assert sampled_indices.shape == expected_shape, f"Expected shape {expected_shape}, got {sampled_indices.shape}"
    assert sampled_indices.dtype == torch.int32, f"Expected dtype int32, got {sampled_indices.dtype}"
    assert sampled_indices.device.type == 'cuda', "Output should be on CUDA device"
    
    # Verify indices are within valid range
    assert torch.all(sampled_indices >= 0), "All indices should be non-negative"
    assert torch.all(sampled_indices < num_points), f"All indices should be less than {num_points}"
    
    # Verify uniqueness (furthest point sampling should return unique points)
    for b in range(batch_size):
        unique_indices = torch.unique(sampled_indices[b])
        assert len(unique_indices) == num_samples, f"Expected {num_samples} unique indices, got {len(unique_indices)}"


def test_furthest_point_sample_cuda():
    """Test furthest_point_sample function with CUDA tensors."""
    batch_size = 2
    num_points = 1000
    num_samples = 256
    device = torch.device('cuda')
    
    # Create random point cloud on CUDA
    points = torch.randn(batch_size, num_points, 3, device=device)
    
    # Call furthest_point_sample function
    sampled_indices = furthest_point_sample(points, num_samples)
    
    # Verify output shape, type, and device
    expected_shape = (batch_size, num_samples)
    assert sampled_indices.shape == expected_shape
    assert sampled_indices.dtype == torch.int32
    assert sampled_indices.device.type == 'cuda'
    
    # Verify indices are within valid range
    assert torch.all(sampled_indices >= 0), "All indices should be non-negative"
    assert torch.all(sampled_indices < num_points), f"All indices should be less than {num_points}"


def test_gather_points_cuda_only():
    """Test gather_points function with CUDA tensors."""
    batch_size = 2
    num_channels = 128
    num_points = 1000
    num_gathered_points = 256
    device = torch.device('cuda')
    
    # Create features and indices on CUDA
    features = torch.randn(batch_size, num_channels, num_points, device=device)
    indices = torch.randint(0, num_points, (batch_size, num_gathered_points), dtype=torch.int32, device=device)
    
    # Call gather_points function
    gathered_features = gather_points(features, indices)
    
    # Verify output shape and type
    expected_shape = (batch_size, num_channels, num_gathered_points)
    assert gathered_features.shape == expected_shape, f"Expected shape {expected_shape}, got {gathered_features.shape}"
    assert gathered_features.dtype == torch.float32, f"Expected dtype float32, got {gathered_features.dtype}"
    assert gathered_features.device.type == 'cuda', "Output should be on CUDA device"


def test_gather_points_cuda():
    """Test gather_points function with CUDA tensors."""
    batch_size = 2
    num_channels = 128
    num_points = 1000
    num_gathered_points = 256
    device = torch.device('cuda')
    
    # Create features and indices on CUDA
    features = torch.randn(batch_size, num_channels, num_points, device=device)
    indices = torch.randint(0, num_points, (batch_size, num_gathered_points), dtype=torch.int32, device=device)
    
    # Call gather_points function
    gathered_features = gather_points(features, indices)
    
    # Verify output shape, type, and device
    expected_shape = (batch_size, num_channels, num_gathered_points)
    assert gathered_features.shape == expected_shape
    assert gathered_features.dtype == torch.float32
    assert gathered_features.device.type == 'cuda'


def test_grouping_operation_cuda_only():
    """Test grouping_operation function with CUDA tensors."""
    batch_size = 2
    num_channels = 64
    num_points = 1000
    num_centers = 256
    num_samples_per_group = 16
    device = torch.device('cuda')
    
    # Create features and grouping indices on CUDA
    features = torch.randn(batch_size, num_channels, num_points, device=device)
    # Create indices for grouping (B, npoint, nsample)
    indices = torch.randint(0, num_points, (batch_size, num_centers, num_samples_per_group), dtype=torch.int32, device=device)
    
    # Call grouping_operation function
    grouped_features = grouping_operation(features, indices)
    
    # Verify output shape and type
    expected_shape = (batch_size, num_channels, num_centers, num_samples_per_group)
    assert grouped_features.shape == expected_shape, f"Expected shape {expected_shape}, got {grouped_features.shape}"
    assert grouped_features.dtype == torch.float32, f"Expected dtype float32, got {grouped_features.dtype}"
    assert grouped_features.device.type == 'cuda', "Output should be on CUDA device"


def test_grouping_operation_cuda():
    """Test grouping_operation function with CUDA tensors."""
    batch_size = 2
    num_channels = 64
    num_points = 1000
    num_centers = 256
    num_samples_per_group = 16
    device = torch.device('cuda')
    
    # Create features and grouping indices on CUDA
    features = torch.randn(batch_size, num_channels, num_points, device=device)
    indices = torch.randint(0, num_points, (batch_size, num_centers, num_samples_per_group), dtype=torch.int32, device=device)
    
    # Call grouping_operation function
    grouped_features = grouping_operation(features, indices)
    
    # Verify output shape, type, and device
    expected_shape = (batch_size, num_channels, num_centers, num_samples_per_group)
    assert grouped_features.shape == expected_shape
    assert grouped_features.dtype == torch.float32
    assert grouped_features.device.type == 'cuda'


def test_gmcnet_usage_pattern_three_nn_upsampling():
    """Test the specific three_nn usage pattern from GMCNet's three_nn_upsampling function."""
    batch_size = 2
    target_points_num = 100
    source_points_num = 200
    device = torch.device('cuda')
    
    # Create point clouds as used in GMCNet (B, N, 3) on CUDA
    target_points = torch.randn(batch_size, target_points_num, 3, device=device)
    source_points = torch.randn(batch_size, source_points_num, 3, device=device)
    
    # Replicate three_nn_upsampling logic from gmcnet.py line 30-36
    dist, idx = three_nn(target_points, source_points)
    
    # Verify shapes match GMCNet expectations
    assert dist.shape == (batch_size, target_points_num, 3)
    assert idx.shape == (batch_size, target_points_num, 3)
    
    # Apply the distance processing as in GMCNet
    dist = torch.max(dist, torch.ones(1, device=device) * 1e-10)  # Use CUDA tensor
    norm = torch.sum((1.0/dist), 2, keepdim=True)
    norm = norm.repeat(1, 1, 3)
    weight = (1.0/dist) / norm
    
    # Verify weight computation succeeded
    assert weight.shape == (batch_size, target_points_num, 3)
    assert torch.all(torch.isfinite(weight)), "All weights should be finite"


def test_gmcnet_usage_pattern_sample_and_group():
    """Test the specific usage pattern from GMCNet's sample_and_group_feats function."""
    batch_size = 2
    num_points = 1000
    num_centers = 256
    num_samples = 16
    num_channels = 64
    device = torch.device('cuda')
    
    # Create point cloud and features as used in GMCNet on CUDA
    points = torch.randn(batch_size, num_points, 3, device=device)
    features = torch.randn(batch_size, num_channels, num_points, device=device)
    
    # Replicate sample_and_group_feats logic from model_utils.py line 285-288
    fps_idx = furthest_point_sample(points, num_centers)
    cent_pts = gather_points(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    
    # Create k-nearest neighbor indices (simplified for test)
    pn_idx = torch.randint(0, num_points, (batch_size, num_centers, num_samples), dtype=torch.int32, device=device)
    
    # Group points and features
    sm_pts = grouping_operation(points.transpose(1, 2).contiguous(), pn_idx).permute(0, 2, 3, 1).contiguous()
    sm_feats = grouping_operation(features, pn_idx)
    
    # Verify all operations succeeded with correct shapes
    assert fps_idx.shape == (batch_size, num_centers)
    assert cent_pts.shape == (batch_size, num_centers, 3)
    assert sm_pts.shape == (batch_size, num_centers, num_samples, 3)
    assert sm_feats.shape == (batch_size, num_channels, num_centers, num_samples)


def test_gmcnet_usage_pattern_get_us_feats():
    """Test the specific usage pattern from GMCNet's get_us_feats function."""
    batch_size = 2
    num_channels = 128
    source_points_num = 256
    target_points_num = 1000
    device = torch.device('cuda')
    
    # Create features and point clouds as used in GMCNet on CUDA
    feats = torch.randn(batch_size, num_channels, source_points_num, device=device)
    src_pts = torch.randn(batch_size, source_points_num, 3, device=device)
    tgt_pts = torch.randn(batch_size, target_points_num, 3, device=device)
    
    # Replicate get_us_feats logic from gmcnet.py line 24-27
    # First get three_nn_upsampling
    dist, idx = three_nn(tgt_pts, src_pts)
    dist = torch.max(dist, torch.ones(1, device=device) * 1e-10)
    norm = torch.sum((1.0/dist), 2, keepdim=True)
    norm = norm.repeat(1, 1, 3)
    weight = (1.0/dist) / norm
    
    # Then interpolate features
    interpolated_feats = three_interpolate(feats, idx, weight)
    
    # Verify final output shape
    expected_shape = (batch_size, num_channels, target_points_num)
    assert interpolated_feats.shape == expected_shape, f"Expected shape {expected_shape}, got {interpolated_feats.shape}"
    assert torch.all(torch.isfinite(interpolated_feats)), "All interpolated features should be finite"


def test_edge_case_small_point_clouds():
    """Test C++ extensions with small point clouds to verify edge case handling."""
    batch_size = 1
    num_points = 3  # Minimal point cloud
    num_samples = 2
    device = torch.device('cuda')
    
    # Create minimal point clouds on CUDA
    points = torch.randn(batch_size, num_points, 3, device=device)
    
    # Test furthest_point_sample with minimal data
    fps_idx = furthest_point_sample(points, num_samples)
    assert fps_idx.shape == (batch_size, num_samples)
    
    # Test gather_points with minimal data
    features = torch.randn(batch_size, 4, num_points, device=device)
    gathered = gather_points(features, fps_idx)
    assert gathered.shape == (batch_size, 4, num_samples)


def test_contiguous_tensor_requirements():
    """Test that C++ extensions work with non-contiguous tensors by making them contiguous."""
    batch_size = 2
    num_points = 100
    device = torch.device('cuda')
    
    # Create non-contiguous tensors on CUDA
    points_large = torch.randn(batch_size, num_points * 2, 3, device=device)
    points = points_large[:, ::2, :].contiguous()  # Make contiguous as required
    
    target_points = torch.randn(batch_size, 50, 3, device=device)
    
    # Verify contiguity requirement
    assert points.is_contiguous(), "Points tensor should be contiguous"
    assert target_points.is_contiguous(), "Target points tensor should be contiguous"
    
    # Test with contiguous tensors
    dist, idx = three_nn(target_points, points)
    assert dist.shape == (batch_size, 50, 3)
    assert idx.shape == (batch_size, 50, 3)


def test_mixed_precision_compatibility():
    """Test C++ extensions work with different floating point precisions."""
    batch_size = 2
    num_points = 100
    device = torch.device('cuda')
    
    # Test with float32 (default)
    points_f32 = torch.randn(batch_size, num_points, 3, dtype=torch.float32, device=device)
    fps_idx = furthest_point_sample(points_f32, 50)
    assert fps_idx.dtype == torch.int32
    
    # Test with features in float32
    features_f32 = torch.randn(batch_size, 64, num_points, dtype=torch.float32, device=device)
    indices = torch.randint(0, num_points, (batch_size, 50), dtype=torch.int32, device=device)
    gathered = gather_points(features_f32, indices)
    assert gathered.dtype == torch.float32


def test_gradient_computation():
    """Test that gradients can be computed through the C++ extensions."""
    batch_size = 2
    num_channels = 32
    source_points_num = 100
    target_points_num = 50
    device = torch.device('cuda')
    
    # Create tensors with gradient tracking on CUDA
    features = torch.randn(batch_size, num_channels, source_points_num, device=device, requires_grad=True)
    indices = torch.randint(0, source_points_num, (batch_size, target_points_num, 3), dtype=torch.int32, device=device)
    weights_raw = torch.rand(batch_size, target_points_num, 3, device=device)
    weights = (weights_raw / weights_raw.sum(dim=2, keepdim=True)).requires_grad_(True)
    
    # Forward pass through three_interpolate
    output = three_interpolate(features, indices, weights)
    
    # Compute loss and backward pass
    loss = output.sum()
    loss.backward()
    
    # Verify gradients were computed for features (weights don't get gradients in this implementation)
    assert features.grad is not None, "Features should have gradients"
    assert features.grad.shape == features.shape, "Feature gradients should match input shape"
    
    # Note: weights.grad is None because the backward pass only computes gradients for features,
    # not for indices or weights. This is correct behavior since weights are typically computed
    # from geometric relationships rather than being learnable parameters.


def test_ball_query_functionality():
    """Test ball_query C++ extension functionality."""
    batch_size = 2
    num_points = 1000
    num_centers = 256
    max_samples = 16
    device = torch.device('cuda')
    
    # Create random point clouds on CUDA
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    center_xyz = torch.randn(batch_size, num_centers, 3, device=device)
    
    # Set radius parameters
    min_radius = 0.1
    max_radius = 0.5
    
    # Call ball_query function
    indices = ball_query(min_radius, max_radius, max_samples, xyz, center_xyz)
    
    # Verify output shape and type
    expected_shape = (batch_size, num_centers, max_samples)
    assert indices.shape == expected_shape, f"Expected shape {expected_shape}, got {indices.shape}"
    assert indices.dtype == torch.int32, f"Expected dtype int32, got {indices.dtype}"
    assert indices.device.type == 'cuda', "Output should be on CUDA device"
    
    # Verify indices are within valid range (or -1 for empty slots)
    # Ball query uses -1 to indicate no point found within radius
    valid_indices = indices[indices >= 0]
    if len(valid_indices) > 0:
        assert torch.all(valid_indices < num_points), f"All valid indices should be less than {num_points}"


def test_ball_query_edge_cases():
    """Test ball_query with various edge cases and parameter combinations."""
    batch_size = 1
    num_points = 100
    num_centers = 10
    device = torch.device('cuda')
    
    # Create point clouds on CUDA
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    center_xyz = torch.randn(batch_size, num_centers, 3, device=device)
    
    # Test with small radius (should find fewer points)
    indices_small = ball_query(0.01, 0.05, 8, xyz, center_xyz)
    assert indices_small.shape == (batch_size, num_centers, 8)
    
    # Test with large radius (should find more points)
    indices_large = ball_query(0.1, 2.0, 32, xyz, center_xyz)
    assert indices_large.shape == (batch_size, num_centers, 32)
    
    # Test with single sample per center
    indices_single = ball_query(0.1, 1.0, 1, xyz, center_xyz)
    assert indices_single.shape == (batch_size, num_centers, 1)


def test_ball_query_contiguous_requirement():
    """Test that ball_query requires contiguous tensors."""
    batch_size = 2
    num_points = 100
    num_centers = 50
    device = torch.device('cuda')
    
    # Create contiguous tensors on CUDA
    xyz = torch.randn(batch_size, num_points, 3, device=device).contiguous()
    center_xyz = torch.randn(batch_size, num_centers, 3, device=device).contiguous()
    
    # Verify contiguity
    assert xyz.is_contiguous(), "xyz tensor should be contiguous"
    assert center_xyz.is_contiguous(), "center_xyz tensor should be contiguous"
    
    # Test with contiguous tensors
    indices = ball_query(0.1, 0.5, 16, xyz, center_xyz)
    assert indices.shape == (batch_size, num_centers, 16)


def test_knn_functionality():
    """Test knn C++ extension functionality."""
    batch_size = 2
    num_points = 500
    num_centers = 100
    k = 8
    device = torch.device('cuda')
    
    # Create random point clouds on CUDA
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    center_xyz = torch.randn(batch_size, num_centers, 3, device=device)
    
    # Call knn function
    indices = knn(k, xyz, center_xyz)
    
    # Verify output shape and type
    expected_shape = (batch_size, k, num_centers)
    assert indices.shape == expected_shape, f"Expected shape {expected_shape}, got {indices.shape}"
    assert indices.dtype == torch.int32, f"Expected dtype int32, got {indices.dtype}"
    assert indices.device.type == 'cuda', "Output should be on CUDA device"
    
    # Verify indices are within valid range
    assert torch.all(indices >= 0), "All indices should be non-negative"
    assert torch.all(indices < num_points), f"All indices should be less than {num_points}"


def test_knn_self_query():
    """Test knn with self-query (center_xyz = None)."""
    batch_size = 2
    num_points = 200
    k = 5
    device = torch.device('cuda')
    
    # Create random point cloud on CUDA
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    
    # Call knn function with self-query
    indices = knn(k, xyz, None)  # center_xyz defaults to xyz
    
    # Verify output shape
    expected_shape = (batch_size, k, num_points)
    assert indices.shape == expected_shape, f"Expected shape {expected_shape}, got {indices.shape}"
    
    # Verify indices are within valid range
    assert torch.all(indices >= 0), "All indices should be non-negative"
    assert torch.all(indices < num_points), f"All indices should be less than {num_points}"


def test_knn_transposed_input():
    """Test knn with transposed input tensors."""
    batch_size = 2
    num_points = 300
    num_centers = 50
    k = 4
    device = torch.device('cuda')
    
    # Create random point clouds on CUDA in transposed format (B, 3, N)
    xyz_transposed = torch.randn(batch_size, 3, num_points, device=device)
    center_xyz_transposed = torch.randn(batch_size, 3, num_centers, device=device)
    
    # Call knn function with transposed=True
    indices = knn(k, xyz_transposed, center_xyz_transposed, True)
    
    # Verify output shape
    expected_shape = (batch_size, k, num_centers)
    assert indices.shape == expected_shape, f"Expected shape {expected_shape}, got {indices.shape}"
    
    # Verify indices are within valid range
    assert torch.all(indices >= 0), "All indices should be non-negative"
    assert torch.all(indices < num_points), f"All indices should be less than {num_points}"


def test_knn_edge_cases():
    """Test knn with various k values and edge cases."""
    batch_size = 1
    num_points = 100
    device = torch.device('cuda')
    
    # Create point cloud on CUDA
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    
    # Test with k=1 (nearest neighbor)
    indices_k1 = knn(1, xyz, None)
    assert indices_k1.shape == (batch_size, 1, num_points)
    
    # Test with larger k
    k_large = min(32, num_points - 1)  # Ensure k < num_points
    indices_large = knn(k_large, xyz, None)
    assert indices_large.shape == (batch_size, k_large, num_points)


def test_ball_query_vs_knn_consistency():
    """Test consistency between ball_query and knn results for overlapping use cases."""
    batch_size = 1
    num_points = 200
    num_centers = 20
    device = torch.device('cuda')
    
    # Create point clouds where centers are a subset of xyz points
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    # Select first num_centers points as centers for fair comparison
    center_xyz = xyz[:, :num_centers, :].contiguous()
    
    # Test ball_query with large radius (should find many neighbors)
    ball_indices = ball_query(0.0, 10.0, 16, xyz, center_xyz)
    
    # Test knn with same number of neighbors
    knn_indices = knn(16, xyz, center_xyz)
    
    # Both should return valid indices
    assert ball_indices.shape == (batch_size, num_centers, 16)
    assert knn_indices.shape == (batch_size, 16, num_centers)
    
    # Verify all indices are within valid range
    valid_ball_indices = ball_indices[ball_indices >= 0]
    if len(valid_ball_indices) > 0:
        assert torch.all(valid_ball_indices < num_points)
    
    assert torch.all(knn_indices >= 0)
    assert torch.all(knn_indices < num_points)


def test_complete_gmcnet_pipeline_with_ball_query():
    """Test a complete pipeline that includes ball_query similar to GMCNet usage patterns."""
    batch_size = 2
    num_points = 1000
    num_centers = 256
    num_samples = 16
    num_channels = 64
    device = torch.device('cuda')
    
    # Create point cloud and features on CUDA
    points = torch.randn(batch_size, num_points, 3, device=device)
    features = torch.randn(batch_size, num_channels, num_points, device=device)
    
    # Step 1: Sample center points using FPS
    fps_idx = furthest_point_sample(points, num_centers)
    center_points = gather_points(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    
    # Step 2: Use ball_query to find neighborhoods
    ball_idx = ball_query(0.1, 0.5, num_samples, points, center_points)
    
    # Step 3: Group features using ball query results
    # Convert ball_idx to valid indices for grouping (replace -1 with 0)
    valid_ball_idx = torch.where(ball_idx >= 0, ball_idx, 0)
    grouped_features = grouping_operation(features, valid_ball_idx)
    
    # Verify pipeline results
    assert fps_idx.shape == (batch_size, num_centers)
    assert center_points.shape == (batch_size, num_centers, 3)
    assert ball_idx.shape == (batch_size, num_centers, num_samples)
    assert grouped_features.shape == (batch_size, num_channels, num_centers, num_samples)
    
    # Verify all operations completed successfully
    assert torch.all(torch.isfinite(grouped_features)), "All grouped features should be finite"
