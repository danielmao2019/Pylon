import pytest
import torch
import numpy as np
from scipy.spatial import KDTree

from metrics.vision_3d import RMSE


def compute_rmse_numpy(transformed, target):
    """Original numpy implementation of RMSE."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(transformed)
    return np.sqrt(np.mean(distances ** 2))


def compute_rmse_with_correspondences_numpy(transformed, target):
    """Original numpy implementation to compute RMSE with correspondences."""
    kdtree = KDTree(target)
    distances, indices = kdtree.query(transformed)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse, indices


def compute_rmse_torch(transformed, target):
    """PyTorch implementation of RMSE."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)

    # Find nearest neighbor distances
    min_distances, min_indices = torch.min(dist_matrix, dim=1)  # (N,), (N,)

    # Compute RMSE
    rmse = torch.sqrt(torch.mean(min_distances ** 2))

    return rmse, min_indices


def test_rmse():
    """Test RMSE calculation."""
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

    # Create RMSE instance
    rmse_metric = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse_metric(source_torch, target_torch)

    # Compute RMSE using NumPy implementation for verification
    numpy_result = compute_rmse_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['rmse'].item()}, NumPy: {numpy_result}"


def test_compute_with_correspondences():
    """Test RMSE with correspondences."""
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

    # Create RMSE instance
    rmse_metric = RMSE()

    # Compute RMSE with correspondences using the metric class
    metric_rmse, metric_indices = rmse_metric.get_correspondences(source_torch, target_torch)

    # Convert PyTorch tensor to numpy array for comparison
    metric_indices_np = metric_indices.cpu().numpy()

    # Compute RMSE with correspondences using NumPy implementation for verification
    numpy_rmse, numpy_indices = compute_rmse_with_correspondences_numpy(source_np, target_np)

    # Check that the RMSE values are approximately equal
    assert abs(metric_rmse.item() - numpy_rmse) < 1e-5, f"Metric RMSE: {metric_rmse.item()}, NumPy RMSE: {numpy_rmse}"

    # Check that the correspondences match
    assert np.array_equal(metric_indices_np, numpy_indices), f"Metric indices: {metric_indices_np}, NumPy indices: {numpy_indices}"


def test_with_random_point_clouds():
    """Test with randomly generated point clouds."""
    # Generate random point clouds
    np.random.seed(42)
    source_np = np.random.randn(100, 3)
    target_np = np.random.randn(150, 3)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create RMSE instance
    rmse_metric = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse_metric(source_torch, target_torch)

    # Compute RMSE using NumPy implementation for verification
    numpy_result = compute_rmse_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['rmse'].item()}, NumPy: {numpy_result}"


def test_known_distance():
    """Test with known distances to verify correctness."""
    # Create point clouds with known distances
    source_np = np.array([
        [0.0, 0.0, 0.0],  # Distance to nearest target: 0.1732
        [1.0, 0.0, 0.0],  # Distance to nearest target: 0.1732
    ])
    target_np = np.array([
        [0.1, 0.1, 0.1],  # Distance from first source: sqrt(0.01+0.01+0.01) = 0.1732
        [1.1, 0.1, 0.1],  # Distance from second source: sqrt(0.01+0.01+0.01) = 0.1732
    ])

    # Expected RMSE: sqrt((0.1732^2 + 0.1732^2)/2) = 0.1732
    expected_rmse = np.sqrt((0.1732**2 + 0.1732**2)/2)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create RMSE instance
    rmse_metric = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse_metric(source_torch, target_torch)

    # Check that the result is approximately equal to expected
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - expected_rmse) < 1e-3, f"Metric: {metric_result['rmse'].item()}, Expected: {expected_rmse}"


def test_rmse_batch():
    """Test RMSE with batched inputs."""
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

    # Create RMSE instance
    rmse_metric = RMSE()

    # Compute RMSE for the batch
    batch_result = rmse_metric(source_torch, target_torch)

    # Verify the shape of the result
    assert batch_result.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {batch_result.shape}"

    # Compute RMSE for each item in the batch using NumPy
    numpy_results = [compute_rmse_numpy(source_np[i], target_np[i]) for i in range(batch_size)]

    # Check that the results are approximately equal
    for i in range(batch_size):
        assert isinstance(batch_result[i], dict), f"{type(batch_result[i])=}"
        assert batch_result[i].keys() == {'rmse'}, f"{batch_result[i].keys()=}"
        assert abs(batch_result[i]['rmse'].item() - numpy_results[i]) < 1e-5, \
            f"Batch {i}: Metric: {batch_result[i]['rmse'].item()}, NumPy: {numpy_results[i]}"
