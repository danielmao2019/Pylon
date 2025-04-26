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
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create a source point cloud with well-separated points
    # This ensures that after translation, each point will still be closest to its translated version
    num_points = 5
    source_np = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [2.5, 2.5, 5.0]
    ])

    # Randomly sample a translation magnitude
    translation_magnitude = np.random.uniform(0.5, 2.0)

    # Generate a random translation direction and normalize it
    translation_direction = np.random.randn(3)
    translation_direction = translation_direction / np.linalg.norm(translation_direction)

    # Apply the translation
    translation = translation_direction * translation_magnitude

    # Create target point cloud by applying translation to source
    target_np = source_np + translation

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create RMSE instance
    rmse = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse(source_torch, target_torch)

    # With well-separated points and a translation that doesn't cause points to cross paths,
    # the RMSE should be exactly equal to the translation magnitude
    expected_rmse = translation_magnitude

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse', 'correspondences'}, f"{metric_result.keys()=}"
    assert abs(metric_result['rmse'].item() - expected_rmse) < 1e-5, \
        f"Metric: {metric_result['rmse'].item()}, Expected: {expected_rmse}"

    # Verify that each point in the source cloud corresponds to its translated point in the target cloud
    # With well-separated points and a small translation, this should be true
    expected_correspondences = torch.arange(num_points, dtype=torch.long)
    assert torch.all(metric_result['correspondences'] == expected_correspondences), \
        f"Metric correspondences: {metric_result['correspondences']}, Expected: {expected_correspondences}"
