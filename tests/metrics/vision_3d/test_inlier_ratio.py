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
    assert result.keys() == {'inlier_ratio', 'inlier_mask', 'inlier_indices'}, \
        f"Expected keys {{'inlier_ratio', 'inlier_mask', 'inlier_indices'}}, got {result.keys()}"
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
    assert metric_result.keys() == {'inlier_ratio', 'inlier_mask', 'inlier_indices'}, \
        f"Expected keys {{'inlier_ratio', 'inlier_mask', 'inlier_indices'}}, got {metric_result.keys()}"
    assert abs(metric_result['inlier_ratio'].item() - numpy_result) < 1e-5, \
        f"Metric: {metric_result['inlier_ratio'].item()}, NumPy: {numpy_result}"


def test_with_known_ratio():
    """Test inlier ratio with synthetic inputs having known ground truth scores."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    num_points = 100
    threshold = 0.5  # Distance threshold for inliers
    noise_scale_inlier = threshold * 0.8  # Noise for inliers (80% of threshold)
    noise_scale_outlier = threshold * 2.0  # Noise for outliers (200% of threshold)
    
    # Create source points with guaranteed minimum separation
    # Use a grid-based approach to ensure points are well-separated
    grid_size = int(np.ceil(np.cbrt(num_points)))
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    z = np.linspace(-1, 1, grid_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    source_np = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    source_np = source_np[:num_points]  # Take only the needed number of points
    
    # Randomly sample a target inlier ratio
    target_ratio = np.random.uniform(0.3, 0.7)
    num_inliers = int(num_points * target_ratio)
    
    # Create target point cloud
    target_np = np.zeros_like(source_np)
    
    # First num_inliers points are inliers (close to source)
    # Add noise that's guaranteed to be less than threshold
    target_np[:num_inliers] = source_np[:num_inliers] + np.random.randn(num_inliers, 3) * noise_scale_inlier
    
    # Remaining points are outliers (far from source)
    # Add noise that's guaranteed to be greater than threshold
    target_np[num_inliers:] = source_np[num_inliers:] + np.random.randn(num_points - num_inliers, 3) * noise_scale_outlier
    
    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    
    # Create InlierRatio instance with the same threshold
    inlier_ratio = InlierRatio(threshold=threshold)
    
    # Compute inlier ratio using the metric class
    metric_result = inlier_ratio(source_torch, target_torch)
    
    # Verify the result
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'inlier_ratio', 'inlier_mask', 'inlier_indices'}, \
        f"Expected keys {{'inlier_ratio', 'inlier_mask', 'inlier_indices'}}, got {metric_result.keys()}"
    
    # The result should be very close to the target ratio since we carefully controlled the noise
    assert abs(metric_result['inlier_ratio'].item() - target_ratio) < 1e-5, \
        f"Metric: {metric_result['inlier_ratio'].item()}, Expected: {target_ratio}"
    
    # Verify that the inlier mask matches our expectations
    inlier_mask = metric_result['inlier_mask']
    assert torch.all(inlier_mask[:num_inliers]), "All points marked as inliers should be inliers"
    assert not torch.any(inlier_mask[num_inliers:]), "All points marked as outliers should be outliers"


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
        assert result.keys() == {'inlier_ratio', 'inlier_mask', 'inlier_indices'}, \
            f"Expected keys {{'inlier_ratio', 'inlier_mask', 'inlier_indices'}}, got {result.keys()}"
        assert abs(result['inlier_ratio'].item() - expected_ratio) < 1e-5, \
            f"Case '{case_name}': Expected {expected_ratio}, got {result['inlier_ratio'].item()}"
