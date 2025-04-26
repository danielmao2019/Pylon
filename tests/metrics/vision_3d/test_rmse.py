import pytest
import torch
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
import math

from metrics.vision_3d import RMSE


def compute_rmse_numpy(source, target):
    """Original numpy implementation of RMSE."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(source)
    return np.sqrt(np.mean(distances ** 2))


def compute_rmse_with_correspondences_numpy(source, target):
    """Original numpy implementation of RMSE with correspondences."""
    kdtree = KDTree(target)
    distances, correspondences = kdtree.query(source)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse, correspondences


@pytest.mark.parametrize("case_name,source,target,expected_rmse,expected_correspondences", [
    ("perfect_match",
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     0.0,
     torch.tensor([0, 1, 2], dtype=torch.long)),
    ("small_offset",
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]], dtype=torch.float32),
     0.17320507764816284,
     torch.tensor([0, 1, 2], dtype=torch.long)),
])
def test_basic_functionality(case_name, source, target, expected_rmse, expected_correspondences):
    """Test basic RMSE calculation with simple examples."""
    rmse = RMSE()
    result = rmse(source, target)
    assert result.keys() == {'rmse', 'correspondences'}, f"Expected keys {{'rmse', 'correspondences'}}, got {result.keys()}"
    assert abs(result['rmse'].item() - expected_rmse) < 1e-5, \
        f"Case '{case_name}': Expected {expected_rmse}, got {result['rmse'].item()}"
    assert torch.all(result['correspondences'] == expected_correspondences), \
        f"Case '{case_name}': Correspondences don't match expected"


def test_with_random_point_clouds():
    """Test RMSE with randomly generated point clouds."""
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
    assert metric_result.keys() == {'rmse', 'correspondences'}, f"Expected keys {{'rmse', 'correspondences'}}, got {metric_result.keys()}"
    assert abs(metric_result['rmse'].item() - numpy_result) < 1e-5, \
        f"Metric: {metric_result['rmse'].item()}, NumPy: {numpy_result}"

    # Get correspondences using NumPy implementation for verification
    _, numpy_correspondences = compute_rmse_with_correspondences_numpy(source_np, target_np)

    # Check that the correspondences match
    assert torch.all(metric_result['correspondences'] == torch.tensor(numpy_correspondences)), \
        f"Metric correspondences: {metric_result['correspondences']}, NumPy correspondences: {numpy_correspondences}"


def test_with_known_rmse():
    """Test RMSE with synthetic inputs having known ground truth scores."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    num_points = 100

    # Randomly sample source points from normal distribution
    source_np = np.random.randn(num_points, 3)

    # Find minimum distance between any pair of points
    min_dist = np.min(pdist(source_np))
    max_translation = min_dist / 2

    # Generate random directions for translation
    directions = np.random.randn(num_points, 3)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    # Generate random distances for translation (less than max_translation)
    distances = np.random.uniform(0, max_translation * 0.99, num_points)  # 99% to be safe
    expected_rmse = np.sqrt(np.mean(distances ** 2))

    # Apply translations to create target point cloud
    target_np = source_np + directions * distances[:, np.newaxis]

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create RMSE instance
    rmse = RMSE()

    # Compute RMSE using the metric class
    metric_result = rmse(source_torch, target_torch)

    # Check that the results are approximately equal to the expected RMSE
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'rmse', 'correspondences'}, f"Expected keys {{'rmse', 'correspondences'}}, got {metric_result.keys()}"
    assert abs(metric_result['rmse'].item() - expected_rmse) < 1e-5, \
        f"Metric: {metric_result['rmse'].item()}, Expected: {expected_rmse}"

    # Verify that each point in the source cloud corresponds to its translated point in the target cloud
    # With well-separated points and a small translation, this should be true
    expected_correspondences = torch.arange(num_points, dtype=torch.long)
    assert torch.all(metric_result['correspondences'] == expected_correspondences), \
        f"Metric correspondences: {metric_result['correspondences']}, Expected: {expected_correspondences}"


@pytest.mark.parametrize("case_name,source,target,expected_rmse,raises_error", [
    ("empty_point_clouds",
     torch.empty((0, 3), dtype=torch.float32),
     torch.empty((0, 3), dtype=torch.float32),
     None,
     IndexError),
    ("single_point",
     torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
     torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
     math.sqrt(3),
     None),
    ("duplicate_points",
     torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32),
     0.0,
     None),
    ("extreme_values",
     torch.tensor([[1e6, 1e6, 1e6], [-1e6, -1e6, -1e6]], dtype=torch.float32),
     torch.tensor([[1e6, 1e6, 1e6], [-1e6, -1e6, -1e6]], dtype=torch.float32),
     0.0,
     None),
    ("nan_values",
     torch.tensor([[0.0, 0.0, 0.0], [float('nan'), float('nan'), float('nan')]], dtype=torch.float32),
     torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
     None,
     AssertionError),
])
def test_edge_cases(case_name, source, target, expected_rmse, raises_error):
    """Test RMSE with edge cases."""
    rmse = RMSE()

    if raises_error:
        with pytest.raises(raises_error):
            rmse(source, target)
    else:
        result = rmse(source, target)
        assert result.keys() == {'rmse', 'correspondences'}, f"Expected keys {{'rmse', 'correspondences'}}, got {result.keys()}"
        assert abs(result['rmse'].item() - expected_rmse) < 1e-5, \
            f"Case '{case_name}': Expected {expected_rmse}, got {result['rmse'].item()}"
