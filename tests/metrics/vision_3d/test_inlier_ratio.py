import pytest
import torch
import numpy as np
from scipy.spatial import KDTree
from metrics.vision_3d import InlierRatio
from scipy.spatial.distance import pdist


def compute_inlier_ratio_numpy(source, target, threshold):
    """Original numpy implementation of inlier ratio."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(source)
    inlier_mask = distances <= threshold
    inlier_ratio = np.mean(inlier_mask)
    inlier_indices = np.where(inlier_mask)[0]
    return {
        'inlier_ratio': inlier_ratio,
        'inlier_mask': inlier_mask,
        'inlier_indices': inlier_indices
    }


@pytest.mark.parametrize("case_name,source,target,threshold,expected_ratio,expected_mask,expected_indices", [
    ("half_inliers", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.float32),
     0.1,
     0.5,
     torch.tensor([True, True, False, False], dtype=torch.bool),
     torch.tensor([0, 1], dtype=torch.long)),
    ("quarter_inliers", 
     torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=torch.float32),
     torch.tensor([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 0.0], [4.0, 4.0, 0.0]], dtype=torch.float32),
     0.1,
     0.25,
     torch.tensor([True, False, False, False], dtype=torch.bool),
     torch.tensor([0], dtype=torch.long)),
])
def test_basic_functionality(case_name, source, target, threshold, expected_ratio, expected_mask, expected_indices):
    """Test basic inlier ratio calculation with simple examples."""
    inlier_ratio = InlierRatio(threshold=threshold)
    result = inlier_ratio(source, target)
    assert result.keys() == {'inlier_ratio', 'inlier_mask', 'inlier_indices'}, \
        f"Expected keys {{'inlier_ratio', 'inlier_mask', 'inlier_indices'}}, got {result.keys()}"
    assert abs(result['inlier_ratio'].item() - expected_ratio) < 1e-5, \
        f"Case '{case_name}': Expected {expected_ratio}, got {result['inlier_ratio'].item()}"
    assert torch.all(result['inlier_mask'] == expected_mask), \
        f"Case '{case_name}': Inlier mask doesn't match expected"
    assert torch.all(result['inlier_indices'] == expected_indices), \
        f"Case '{case_name}': Inlier indices don't match expected"


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
    assert abs(metric_result['inlier_ratio'].item() - numpy_result['inlier_ratio']) < 1e-5, \
        f"Metric: {metric_result['inlier_ratio'].item()}, NumPy: {numpy_result['inlier_ratio']}"
    assert torch.all(metric_result['inlier_mask'] == torch.tensor(numpy_result['inlier_mask'])), \
        "Inlier masks don't match"
    assert torch.all(metric_result['inlier_indices'] == torch.tensor(numpy_result['inlier_indices'])), \
        "Inlier indices don't match"


def test_with_known_ratio():
    """Test inlier ratio with synthetic inputs having known ground truth scores."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    num_points = 1000
    
    # Randomly sample source points from normal distribution
    source_np = np.random.randn(num_points, 3)
    
    # Find minimum distance between any pair of points
    min_dist = np.min(pdist(source_np))
    threshold = min_dist / 2
    
    # Randomly sample a target inlier ratio
    target_ratio = np.random.uniform(0.3, 0.7)
    num_inliers = int(num_points * target_ratio)
    
    # Create target points and ground truth inlier mask/indices
    target_np = np.zeros_like(source_np)
    inlier_mask = np.zeros(num_points, dtype=bool)
    inlier_indices = np.arange(num_inliers)  # First num_inliers points are inliers
    inlier_mask[:num_inliers] = True
    
    # For inliers: apply translation with magnitude < min_dist / 2
    max_translation = min_dist / 2
    # Generate random directions
    directions = np.random.randn(num_inliers, 3)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    # Generate random distances
    distances = np.random.uniform(0, max_translation * 0.99, num_inliers)  # 99% to be safe
    # Apply translations
    target_np[:num_inliers] = source_np[:num_inliers] + directions * distances[:, np.newaxis]
    
    # For outliers: place points outside the source point cloud bounds plus threshold
    source_min = source_np.min(axis=0) - threshold
    source_max = source_np.max(axis=0) + threshold
    # Generate random points outside the bounds
    for i in range(num_inliers, num_points):
        while True:
            # Sample a random point
            point = np.random.randn(3)
            # Check if it's outside the bounds
            if (np.any(point < source_min) or np.any(point > source_max)):
                target_np[i] = point
                break
    
    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    expected_mask = torch.tensor(inlier_mask, dtype=torch.bool)
    expected_indices = torch.tensor(inlier_indices, dtype=torch.long)
    
    # Create InlierRatio instance with the same threshold
    inlier_ratio = InlierRatio(threshold=threshold)
    
    # Compute inlier ratio using the metric class
    metric_result = inlier_ratio(source_torch, target_torch)
    
    # Verify the result
    assert isinstance(metric_result, dict), f"{type(metric_result)=}"
    assert metric_result.keys() == {'inlier_ratio', 'inlier_mask', 'inlier_indices'}, \
        f"Expected keys {{'inlier_ratio', 'inlier_mask', 'inlier_indices'}}, got {metric_result.keys()}"
    
    # The result should be exactly equal to the target ratio
    assert abs(metric_result['inlier_ratio'].item() - target_ratio) < 1/num_points, \
        f"Metric: {metric_result['inlier_ratio'].item()}, Expected: {target_ratio}"
    
    # Verify that the inlier mask and indices match our expectations
    assert torch.equal(metric_result['inlier_mask'], expected_mask), \
        f"{metric_result['inlier_mask']=}, {expected_mask=}"
    assert torch.equal(metric_result['inlier_indices'], expected_indices), \
        f"{metric_result['inlier_indices']=}, {expected_indices=}"


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
