import pytest
import math
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist

from metrics.vision_3d import MAE


def compute_mae_numpy(source, target):
    """Alternative numpy implementation of MAE using KDTree."""
    # Build KD-trees for both point clouds
    kdtree = KDTree(target)

    # Find nearest neighbors from source to target
    distances, _ = kdtree.query(source)

    # Compute MAE as average of both directions
    mae = np.mean(distances)
    return mae


@pytest.mark.parametrize("case_name,source,target,expected_mae", [
    ("perfect_match",
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     0.0),
    ("small_offset",
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]], dtype=torch.float32),
     0.17320507764816284),
])
def test_basic_functionality(case_name, source, target, expected_mae):
    """Test basic MAE calculation with simple examples."""
    mae = MAE()
    result = mae(source, target)
    assert result.keys() == {'mae'}, f"Expected keys {{'mae'}}, got {result.keys()}"
    assert abs(result['mae'].item() - expected_mae) < 1e-5, \
        f"Case '{case_name}': Expected {expected_mae}, got {result['mae'].item()}"


def test_with_random_point_clouds():
    """Test MAE with randomly generated point clouds."""
    # Generate random point clouds
    np.random.seed(42)
    source_np = np.random.randn(100, 3)
    target_np = np.random.randn(150, 3)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create MAE instance
    mae = MAE()

    # Compute MAE using the metric class
    metric_result = mae(source_torch, target_torch)

    # Compute MAE using NumPy implementation for verification
    numpy_result = compute_mae_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'mae'}, f"Expected keys {{'mae'}}, got {metric_result.keys()}"
    assert abs(metric_result['mae'].item() - numpy_result) < 1e-5, \
        f"Metric: {metric_result['mae'].item()}, NumPy: {numpy_result}"


def test_with_known_mae():
    """Test MAE with synthetic inputs having known ground truth scores."""
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
    expected_mae = np.mean(distances)

    # Apply translations to create target point cloud
    target_np = source_np + directions * distances[:, np.newaxis]

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create MAE instance
    mae = MAE()

    # Compute MAE using the metric class
    metric_result = mae(source_torch, target_torch)

    # Check that the results are approximately equal to the expected MAE
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'mae'}, f"Expected keys {{'mae'}}, got {metric_result.keys()}"
    assert abs(metric_result['mae'].item() - expected_mae) < 1e-5, \
        f"Metric: {metric_result['mae'].item()}, Expected: {expected_mae}"


@pytest.mark.parametrize("case_name,source,target,expected_mae,raises_error", [
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
def test_edge_cases(case_name, source, target, expected_mae, raises_error):
    """Test MAE with edge cases."""
    mae = MAE()

    if raises_error:
        with pytest.raises(raises_error):
            mae(source, target)
    else:
        result = mae(source, target)
        assert result.keys() == {'mae'}, f"Expected keys {{'mae'}}, got {result.keys()}"
        assert abs(result['mae'].item() - expected_mae) < 1e-5, \
            f"Case '{case_name}': Expected {expected_mae}, got {result['mae'].item()}"
