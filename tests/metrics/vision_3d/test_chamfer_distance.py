import pytest
import torch
import numpy as np
from metrics.vision_3d import ChamferDistance
from scipy.spatial import KDTree


def create_datapoint(outputs, labels, idx=0):
    """Helper function to create datapoint with proper structure."""
    return {
        'inputs': {},
        'outputs': outputs,
        'labels': labels, 
        'meta_info': {'idx': idx}
    }


def compute_chamfer_distance_numpy(source, target):
    """Alternative numpy implementation of Chamfer distance using KDTree."""
    # Build KD-trees for both point clouds
    source_tree = KDTree(source)
    target_tree = KDTree(target)

    # Find nearest neighbors from source to target
    source_to_target_dist, _ = source_tree.query(target)
    # Find nearest neighbors from target to source
    target_to_source_dist, _ = target_tree.query(source)

    # Compute Chamfer distance as sum of squared distances in both directions
    chamfer_distance = np.mean(source_to_target_dist) + np.mean(target_to_source_dist)
    return chamfer_distance


@pytest.mark.parametrize("case_name,source,target,expected_distance", [
    ("perfect_match",
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     0.0),
    ("small_offset",
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]], dtype=torch.float32),
     0.3464101552963257),
])
def test_basic_functionality(case_name, source, target, expected_distance):
    """Test basic Chamfer distance calculation with simple examples."""
    chamfer = ChamferDistance()
    datapoint = create_datapoint(source, target)
    result = chamfer(datapoint)
    assert result.keys() == {'chamfer_distance'}, f"Expected keys {{'chamfer_distance'}}, got {result.keys()}"
    assert abs(result['chamfer_distance'].item() - expected_distance) < 1e-5, \
        f"Case '{case_name}': Expected {expected_distance}, got {result['chamfer_distance'].item()}"


def test_with_random_point_clouds():
    """Test Chamfer distance with randomly generated point clouds."""
    # Generate random point clouds
    np.random.seed(42)
    source_np = np.random.randn(100, 3)
    target_np = np.random.randn(150, 3)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create ChamferDistance instance
    chamfer = ChamferDistance()

    # Compute Chamfer distance using the metric class
    datapoint = create_datapoint(source_torch, target_torch)
    metric_result = chamfer(datapoint)

    # Compute Chamfer distance using NumPy implementation for verification
    numpy_result = compute_chamfer_distance_numpy(source_np, target_np)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'chamfer_distance'}, f"Expected keys {{'chamfer_distance'}}, got {metric_result.keys()}"
    assert abs(metric_result['chamfer_distance'].item() - numpy_result) < 1e-5, \
        f"Metric: {metric_result['chamfer_distance'].item()}, NumPy: {numpy_result}"


def test_with_known_distance():
    """Test Chamfer distance with synthetic inputs having known ground truth scores."""
    # Create a source point cloud with well-separated points
    # Use a grid-like pattern to ensure points are well-separated
    x = np.linspace(-1, 1, 5)
    y = np.linspace(-1, 1, 5)
    z = np.linspace(-1, 1, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    source_np = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

    # Apply a known translation that preserves correspondence
    # Since points are well-separated, a small translation won't change nearest neighbors
    translation = np.array([0.2, 0.2, 0.2])
    target_np = source_np + translation

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create ChamferDistance instance
    chamfer = ChamferDistance()

    # Compute Chamfer distance using the metric class
    datapoint = create_datapoint(source_torch, target_torch)
    metric_result = chamfer(datapoint)

    # Expected distance is twice the translation magnitude (bidirectional)
    expected_distance = 2 * np.linalg.norm(translation)

    # Check that the results are approximately equal to the expected distance
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'chamfer_distance'}, f"Expected keys {{'chamfer_distance'}}, got {metric_result.keys()}"
    assert abs(metric_result['chamfer_distance'].item() - expected_distance) < 1e-3, \
        f"Metric: {metric_result['chamfer_distance'].item()}, Expected: {expected_distance}"


@pytest.mark.parametrize("case_name,source,target,expected_distance,raises_error", [
    ("empty_point_clouds",
     torch.empty((0, 3), dtype=torch.float32),
     torch.empty((0, 3), dtype=torch.float32),
     None,
     IndexError),
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
     AssertionError),
])
def test_edge_cases(case_name, source, target, expected_distance, raises_error):
    """Test Chamfer distance with edge cases."""
    chamfer = ChamferDistance()

    if raises_error:
        with pytest.raises(raises_error):
            datapoint = create_datapoint(source, target)
            chamfer(datapoint)
    else:
        datapoint = create_datapoint(source, target)
        result = chamfer(datapoint)
        assert result.keys() == {'chamfer_distance'}, f"Expected keys {{'chamfer_distance'}}, got {result.keys()}"
        assert abs(result['chamfer_distance'].item() - expected_distance) < 1e-5, \
            f"Case '{case_name}': Expected {expected_distance}, got {result['chamfer_distance'].item()}"
