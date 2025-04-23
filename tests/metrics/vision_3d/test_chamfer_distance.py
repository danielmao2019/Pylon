import torch
import numpy as np
from scipy.spatial import KDTree

from metrics.vision_3d import ChamferDistance


def compute_chamfer_distance_numpy(transformed, target):
    """Original numpy implementation of chamfer distance."""
    kdtree_source = KDTree(transformed)
    kdtree_target = KDTree(target)

    # Distance from transformed to target
    dist_source_to_target, _ = kdtree_target.query(transformed)

    # Distance from target to transformed
    dist_target_to_source, _ = kdtree_source.query(target)

    # The Chamfer Distance is the sum of means
    return np.mean(dist_source_to_target) + np.mean(dist_target_to_source)


def compute_chamfer_distance_unidirectional_numpy(source, target):
    """Original numpy implementation of unidirectional chamfer distance."""
    kdtree_target = KDTree(target)
    dist_source_to_target, _ = kdtree_target.query(source)
    return np.mean(dist_source_to_target)


def test_chamfer_distance_bidirectional():
    """Test bidirectional Chamfer distance implementation."""
    # Create sample point clouds
    source_np = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    target_np = np.array([
        [0.1, 0.1, 0.1],
        [1.1, 0.1, 0.1],
        [0.1, 1.1, 0.1],
        [1.1, 1.1, 0.1],
        [0.5, 0.5, 0.5]  # Additional point
    ])

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create ChamferDistance instance
    chamfer_distance = ChamferDistance(bidirectional=True)

    # Compute Chamfer distance using the metric class
    metric_result = chamfer_distance(source_torch, target_torch)

    # Compute Chamfer distance using NumPy implementation for verification
    numpy_result = compute_chamfer_distance_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'chamfer_distance'}, f"{metric_result.keys()=}"
    assert abs(metric_result['chamfer_distance'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['chamfer_distance'].item()}, NumPy: {numpy_result}"


def test_chamfer_distance_unidirectional():
    """Test unidirectional Chamfer distance implementation."""
    # Create sample point clouds
    source_np = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    target_np = np.array([
        [0.1, 0.1, 0.1],
        [1.1, 0.1, 0.1],
        [0.1, 1.1, 0.1],
        [1.1, 1.1, 0.1],
        [0.5, 0.5, 0.5]  # Additional point
    ])

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create ChamferDistance instance with unidirectional=True
    chamfer_distance = ChamferDistance(bidirectional=False)

    # Compute unidirectional Chamfer distance using the metric class
    metric_result = chamfer_distance(source_torch, target_torch)

    # Compute unidirectional Chamfer distance using NumPy implementation for verification
    numpy_result = compute_chamfer_distance_unidirectional_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'chamfer_distance'}, f"{metric_result.keys()=}"
    assert abs(metric_result['chamfer_distance'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['chamfer_distance'].item()}, NumPy: {numpy_result}"


def test_with_random_point_clouds():
    """Test with randomly generated point clouds."""
    # Generate random point clouds of different sizes
    np.random.seed(42)
    source_np = np.random.randn(100, 3)
    target_np = np.random.randn(150, 3)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create ChamferDistance instance
    chamfer_distance = ChamferDistance(bidirectional=True)

    # Compute Chamfer distance using the metric class
    metric_result = chamfer_distance(source_torch, target_torch)

    # Compute Chamfer distance using NumPy implementation for verification
    numpy_result = compute_chamfer_distance_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'chamfer_distance'}, f"{metric_result.keys()=}"
    assert abs(metric_result['chamfer_distance'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['chamfer_distance'].item()}, NumPy: {numpy_result}"


def test_chamfer_distance_batch():
    """Test Chamfer distance with batched inputs."""
    # Create batch of point clouds
    batch_size = 3
    source_points = 100
    target_points = 150

    # Generate random point clouds
    np.random.seed(42)
    source_np = np.random.randn(batch_size, source_points, 3)
    target_np = np.random.randn(batch_size, target_points, 3)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create ChamferDistance instance
    chamfer = ChamferDistance()

    # Compute Chamfer distance for the batch
    batch_result = chamfer(source_torch, target_torch)

    # Compute Chamfer distance for each item in the batch using NumPy
    numpy_results = [compute_chamfer_distance_numpy(source_np[i], target_np[i]) for i in range(batch_size)]
    
    # Calculate the mean across the batch
    numpy_mean = np.mean(numpy_results)

    # Check that the results are approximately equal
    assert isinstance(batch_result, dict), f"{type(batch_result)=}"
    assert batch_result.keys() == {'chamfer_distance'}, f"{batch_result.keys()=}"
    assert abs(batch_result['chamfer_distance'].item() - numpy_mean) < 1e-5, \
        f"Metric: {batch_result['chamfer_distance'].item()}, NumPy mean: {numpy_mean}"
