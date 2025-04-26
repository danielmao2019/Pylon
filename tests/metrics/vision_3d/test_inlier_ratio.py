import pytest
import torch
import numpy as np
from scipy.spatial import KDTree
from metrics.vision_3d import InlierRatio


def compute_inlier_ratio_numpy(source, target, threshold):
    """Original numpy implementation of inlier ratio."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(source)
    inliers = distances <= threshold
    return np.mean(inliers)


@pytest.mark.parametrize("case_name,source,target,threshold,expected_ratio", [
    ("half_inliers", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.float32),
     0.1,
     0.5),
    ("quarter_inliers", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [4.0, 4.0, 0.0]], dtype=torch.float32),
     0.1,
     0.25),
])
def test_basic_functionality(case_name, source, target, threshold, expected_ratio):
    """Test basic inlier ratio calculation with simple examples."""
    inlier_ratio = InlierRatio(threshold=threshold)
    result = inlier_ratio(source, target)
    assert result.keys() == {'inlier_ratio'}, f"Expected keys {{'inlier_ratio'}}, got {result.keys()}"
    assert abs(result['inlier_ratio'].item() - expected_ratio) < 1e-5, \
        f"Case '{case_name}': Expected {expected_ratio}, got {result['inlier_ratio'].item()}"


def test_with_random_point_clouds():
    """Test inlier ratio with randomly generated point clouds."""
    # Generate random point clouds
    np.random.seed(42)
    source_np = np.random.randn(100, 3)
    target_np = np.random.randn(150, 3)

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create InlierRatio instance
    inlier_ratio = InlierRatio(threshold=0.5)

    # Compute inlier ratio using the metric class
    metric_result = inlier_ratio(source_torch, target_torch)

    # Compute inlier ratio using NumPy implementation for verification
    numpy_result = compute_inlier_ratio_numpy(source_np, target_np, 0.5)

    # Check that the results are approximately equal
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'inlier_ratio'}, f"Expected keys {{'inlier_ratio'}}, got {metric_result.keys()}"
    assert abs(metric_result['inlier_ratio'].item() - numpy_result) < 1e-5, \
        f"Metric: {metric_result['inlier_ratio'].item()}, NumPy: {numpy_result}"


def test_with_known_ratio():
    """Test inlier ratio with synthetic inputs having known ground truth scores."""
    # Create a source point cloud with well-separated points
    num_points = 100
    source_np = np.random.randn(num_points, 3)

    # Randomly sample a target inlier ratio
    target_ratio = np.random.uniform(0.3, 0.7)

    # Create target point cloud with known inlier ratio
    target_np = np.zeros_like(source_np)
    num_inliers = int(num_points * target_ratio)
    
    # First num_inliers points are inliers (close to source)
    target_np[:num_inliers] = source_np[:num_inliers] + np.random.randn(num_inliers, 3) * 0.1
    
    # Remaining points are outliers (far from source)
    target_np[num_inliers:] = source_np[num_inliers:] + np.random.randn(num_points - num_inliers, 3) * 10.0

    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)

    # Create InlierRatio instance
    inlier_ratio = InlierRatio(threshold=0.5)

    # Compute inlier ratio using the metric class
    metric_result = inlier_ratio(source_torch, target_torch)

    # Check that the results are approximately equal to the target ratio
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'inlier_ratio'}, f"Expected keys {{'inlier_ratio'}}, got {metric_result.keys()}"
    assert abs(metric_result['inlier_ratio'].item() - target_ratio) < 0.1, \
        f"Metric: {metric_result['inlier_ratio'].item()}, Expected: {target_ratio}"


@pytest.mark.parametrize("case_name,source,target,expected_ratio,raises_error", [
    ("empty_point_clouds", 
     torch.empty((0, 3), dtype=torch.float32), 
     torch.empty((0, 3), dtype=torch.float32), 
     None, 
     ValueError),
    ("single_point_outlier", 
     torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32), 
     torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32), 
     0.0, 
     None),
    ("all_outliers", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32), 
     torch.tensor([[10.0, 10.0, 10.0], [11.0, 10.0, 10.0], [10.0, 11.0, 10.0]], dtype=torch.float32), 
     0.0, 
     None),
    ("all_inliers", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32), 
     torch.tensor([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1], [0.1, 1.1, 0.1]], dtype=torch.float32), 
     1.0, 
     None),
    ("nan_values", 
     torch.tensor([[0.0, 0.0, 0.0], [float('nan'), float('nan'), float('nan')]], dtype=torch.float32), 
     torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=torch.float32), 
     None, 
     ValueError),
])
def test_edge_cases(case_name, source, target, expected_ratio, raises_error):
    """Test inlier ratio with edge cases."""
    inlier_ratio = InlierRatio(threshold=0.5)
    
    if raises_error:
        with pytest.raises(raises_error):
            inlier_ratio(source, target)
    else:
        result = inlier_ratio(source, target)
        assert result.keys() == {'inlier_ratio'}, f"Expected keys {{'inlier_ratio'}}, got {result.keys()}"
        assert abs(result['inlier_ratio'].item() - expected_ratio) < 1e-5, \
            f"Case '{case_name}': Expected {expected_ratio}, got {result['inlier_ratio'].item()}"
