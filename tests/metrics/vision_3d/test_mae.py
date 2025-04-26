import pytest
import torch
import numpy as np
from scipy.spatial import KDTree

from metrics.vision_3d import MAE


def compute_mae_numpy(source, target):
    """Alternative numpy implementation of MAE using KDTree."""
    # Build KD-trees for both point clouds
    source_tree = KDTree(source)
    target_tree = KDTree(target)
    
    # Find nearest neighbors from source to target
    source_to_target_dist, _ = source_tree.query(target)
    # Find nearest neighbors from target to source
    target_to_source_dist, _ = target_tree.query(source)
    
    # Compute MAE as average of both directions
    mae = (np.mean(source_to_target_dist) + np.mean(target_to_source_dist)) / 2.0
    return mae


def compute_mae_torch(transformed, target):
    """PyTorch implementation of MAE."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)

    # Find nearest neighbor distances
    min_distances = torch.min(dist_matrix, dim=1)[0]  # (N,)

    # Compute MAE
    return torch.mean(min_distances)


@pytest.mark.parametrize("case_name,source,target,expected_mae", [
    ("perfect_match", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     0.0),
    ("small_offset", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]], dtype=torch.float32),
     0.3464101552963257),
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
    # Create a source point cloud with well-separated points
    num_points = 100
    source_np = np.random.randn(num_points, 3)

    # Create target point cloud with known distance
    target_np = source_np + np.random.randn(num_points, 3) * 0.5

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create MAE instance
    mae = MAE()

    # Compute MAE using the metric class
    metric_result = mae(source_torch, target_torch)

    # Compute expected MAE (approximately 0.5^2 = 0.25 for each point)
    expected_mae = 0.25

    # Check that the results are approximately equal to the expected MAE
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'mae'}, f"Expected keys {{'mae'}}, got {metric_result.keys()}"
    assert abs(metric_result['mae'].item() - expected_mae) < 0.1, \
        f"Metric: {metric_result['mae'].item()}, Expected: {expected_mae}"


@pytest.mark.parametrize("case_name,source,target,expected_mae,raises_error", [
    ("empty_point_clouds", 
     torch.empty((0, 3), dtype=torch.float32), 
     torch.empty((0, 3), dtype=torch.float32), 
     None, 
     ValueError),
    ("single_point", 
     torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32), 
     torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32), 
     3.464101552963257, 
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
     ValueError),
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
