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
    """Original numpy implementation of RMSE with correspondences."""
    kdtree = KDTree(target)
    distances, correspondences = kdtree.query(transformed)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse, correspondences


def compute_rmse_torch(transformed, target):
    """PyTorch implementation of RMSE."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)

    # Find nearest neighbor distances
    min_distances = torch.min(dist_matrix, dim=1)[0]  # (N,)

    # Compute RMSE
    return torch.sqrt(torch.mean(min_distances ** 2))


def compute_rmse_with_correspondences_torch(transformed, target):
    """PyTorch implementation of RMSE with correspondences."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)

    # Find nearest neighbor distances and indices
    min_distances, correspondences = torch.min(dist_matrix, dim=1)  # (N,)

    # Compute RMSE
    rmse = torch.sqrt(torch.mean(min_distances ** 2))

    return rmse, correspondences


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
        [0.01, 0.01, 0.01],
        [1.01, 0.01, 0.01],
        [0.01, 1.01, 0.01],
        [1.01, 1.01, 0.01],
        [0.5, 0.5, 0.5]
    ])

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create RMSE instance
    rmse = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse(source_torch, target_torch)

    # Compute RMSE using NumPy implementation for verification
    numpy_result = compute_rmse_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse', 'correspondences'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['rmse'].item()}, NumPy: {numpy_result}"

    # Get correspondences using NumPy implementation for verification
    _, numpy_correspondences = compute_rmse_with_correspondences_numpy(source_np, target_np)

    # Check that the correspondences match
    assert torch.all(metric_result['correspondences'] == torch.tensor(numpy_correspondences)), \
        f"Metric correspondences: {metric_result['correspondences']}, NumPy correspondences: {numpy_correspondences}"


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
    rmse = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse(source_torch, target_torch)

    # Compute RMSE using NumPy implementation for verification
    numpy_result = compute_rmse_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse', 'correspondences'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - numpy_result) < 1e-5, f"Metric: {metric_result['rmse'].item()}, NumPy: {numpy_result}"

    # Get correspondences using NumPy implementation for verification
    _, numpy_correspondences = compute_rmse_with_correspondences_numpy(source_np, target_np)

    # Check that the correspondences match
    assert torch.all(metric_result['correspondences'] == torch.tensor(numpy_correspondences)), \
        f"Metric correspondences: {metric_result['correspondences']}, NumPy correspondences: {numpy_correspondences}"


def test_known_distance():
    """Test with known distances between points."""
    # Create point clouds with known distances
    source_np = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0]
    ])
    target_np = np.array([
        [0.0, 0.0, 0.0],  # Distance: 0.0
        [1.0, 0.0, 0.0],  # Distance: 0.0
        [0.0, 1.0, 0.0],  # Distance: 0.0
        [1.0, 1.0, 0.0],  # Distance: 0.0
        [2.0, 2.0, 0.0],  # Distance: sqrt(2)
    ])

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create RMSE instance
    rmse = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse(source_torch, target_torch)

    # Expected RMSE: sqrt((0^2 + 0^2 + 0^2 + 0^2 + (sqrt(2))^2) / 5) = sqrt(2/5)
    expected_rmse = np.sqrt(2/5)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse', 'correspondences'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - expected_rmse) < 1e-5, \
        f"Metric: {metric_result['rmse'].item()}, Expected: {expected_rmse}"

    # Get correspondences using NumPy implementation for verification
    _, numpy_correspondences = compute_rmse_with_correspondences_numpy(source_np, target_np)

    # Check that the correspondences match
    assert torch.all(metric_result['correspondences'] == torch.tensor(numpy_correspondences)), \
        f"Metric correspondences: {metric_result['correspondences']}, NumPy correspondences: {numpy_correspondences}"


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
    rmse = RMSE()

    # Compute RMSE for the batch
    batch_result = rmse(source_torch, target_torch)

    # Compute RMSE for each item in the batch using NumPy
    numpy_results = [compute_rmse_numpy(source_np[i], target_np[i]) for i in range(batch_size)]
    
    # Calculate the mean across the batch
    numpy_mean = np.mean(numpy_results)

    # Check that the results are approximately equal
    assert isinstance(batch_result, dict), f"{type(batch_result)=}"
    assert batch_result.keys() == {'rmse', 'correspondences'}, f"{batch_result.keys()=}"
    assert abs(batch_result['rmse'].item() - numpy_mean) < 1e-5, \
        f"Metric: {batch_result['rmse'].item()}, NumPy mean: {numpy_mean}"

    # Get correspondences using NumPy implementation for verification
    # We'll check the first batch item's correspondences
    _, numpy_correspondences = compute_rmse_with_correspondences_numpy(source_np[0], target_np[0])

    # Check that the correspondences match for the first batch item
    assert torch.all(batch_result['correspondences'][0] == torch.tensor(numpy_correspondences)), \
        f"Metric correspondences for first batch: {batch_result['correspondences'][0]}, NumPy correspondences: {numpy_correspondences}"
