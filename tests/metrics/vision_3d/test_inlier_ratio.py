import pytest
import torch
import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, List

from metrics.vision_3d import InlierRatio


def compute_inlier_ratio_numpy(transformed, target, threshold=0.02):
    """Original numpy implementation of inlier ratio."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(transformed)
    return np.mean(distances < threshold)


def identify_inliers_numpy(transformed, target, threshold=0.02):
    """Original numpy implementation to identify inliers."""
    kdtree = KDTree(target)
    distances, correspondences = kdtree.query(transformed)
    
    # Boolean mask of inliers
    inlier_mask = distances < threshold
    
    # Get indices of inliers
    inlier_indices = [i for i in range(len(inlier_mask)) if inlier_mask[i]]
    
    return inlier_mask, inlier_indices


def compute_inlier_ratio_torch(transformed, target, threshold=0.02):
    """PyTorch implementation of inlier ratio."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)
    
    # Find nearest neighbor distances
    min_distances = torch.min(dist_matrix, dim=1)[0]  # (N,)
    
    # Count inliers
    inliers = (min_distances < threshold).float()
    return torch.mean(inliers)


def identify_inliers_torch(transformed, target, threshold=0.02):
    """PyTorch implementation to identify inliers."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)
    
    # Find nearest neighbor distances
    min_distances, _ = torch.min(dist_matrix, dim=1)  # (N,)
    
    # Identify inliers
    inlier_mask = min_distances < threshold
    inlier_indices = inlier_mask.nonzero().squeeze().tolist()
    
    # Convert single index to list for consistency
    if isinstance(inlier_indices, int):
        inlier_indices = [inlier_indices]
        
    return inlier_mask, inlier_indices


class TestInlierRatio:
    def test_inlier_ratio(self):
        """Test inlier ratio calculation."""
        # Create sample point clouds with known inliers
        source_np = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        target_np = np.array([
            [0.01, 0.01, 0.01],  # Within threshold
            [1.03, 0.01, 0.01],  # Outside threshold
            [0.01, 1.01, 0.01],  # Within threshold
            [1.01, 1.01, 0.01],  # Within threshold
            [0.5, 0.5, 0.5]      # Far from any source point
        ])
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Set threshold
        threshold = 0.02
        
        # Compute inlier ratio using PyTorch implementation
        torch_result = compute_inlier_ratio_torch(source_torch, target_torch, threshold)
        
        # Compute inlier ratio using NumPy implementation
        numpy_result = compute_inlier_ratio_numpy(source_np, target_np, threshold)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
    
    def test_get_inliers(self):
        """Test identifying inliers."""
        # Create sample point clouds with known inliers
        source_np = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        target_np = np.array([
            [0.01, 0.01, 0.01],  # Within threshold
            [1.03, 0.01, 0.01],  # Outside threshold
            [0.01, 1.01, 0.01],  # Within threshold
            [1.01, 1.01, 0.01],  # Within threshold
            [0.5, 0.5, 0.5]      # Far from any source point
        ])
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Set threshold
        threshold = 0.02
        
        # Get inliers using PyTorch implementation
        torch_mask, torch_indices = identify_inliers_torch(source_torch, target_torch, threshold)
        
        # Convert PyTorch mask to numpy for comparison
        torch_mask_np = torch_mask.cpu().numpy()
        
        # Get inliers using NumPy implementation
        numpy_mask, numpy_indices = identify_inliers_numpy(source_np, target_np, threshold)
        
        # Check that the masks are equal
        assert np.array_equal(torch_mask_np, numpy_mask), f"PyTorch mask: {torch_mask_np}, NumPy mask: {numpy_mask}"
        
        # Check that the indices match
        assert sorted(torch_indices) == sorted(numpy_indices), f"PyTorch indices: {torch_indices}, NumPy indices: {numpy_indices}"
    
    def test_with_random_point_clouds(self):
        """Test with randomly generated point clouds."""
        # Generate random point clouds
        np.random.seed(42)
        source_np = np.random.randn(100, 3)
        target_np = np.random.randn(150, 3)
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Set threshold
        threshold = 0.5  # Larger threshold for random points
        
        # Compute inlier ratio using PyTorch implementation
        torch_result = compute_inlier_ratio_torch(source_torch, target_torch, threshold)
        
        # Compute inlier ratio using NumPy implementation
        numpy_result = compute_inlier_ratio_numpy(source_np, target_np, threshold)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
        
    def test_various_thresholds(self):
        """Test with various threshold values."""
        # Create sample point clouds
        source_np = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        target_np = np.array([
            [0.02, 0.02, 0.02],  # Distance: ~0.0346
            [1.05, 0.05, 0.05],  # Distance: ~0.0866
            [0.05, 1.05, 0.05],  # Distance: ~0.0866
            [1.05, 1.05, 0.05],  # Distance: ~0.0866
        ])
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Test different thresholds
        thresholds = [0.03, 0.05, 0.1]
        expected_ratios = [0.0, 0.25, 1.0]  # Expected results for each threshold
        
        for threshold, expected in zip(thresholds, expected_ratios):
            # Compute inlier ratio using PyTorch implementation
            torch_result = compute_inlier_ratio_torch(source_torch, target_torch, threshold)
            
            # Compute inlier ratio using NumPy implementation
            numpy_result = compute_inlier_ratio_numpy(source_np, target_np, threshold)
            
            # Check that the results match expected value
            assert abs(torch_result.item() - expected) < 1e-5, \
                f"Threshold {threshold}: PyTorch: {torch_result.item()}, Expected: {expected}"
            
            # Check that PyTorch and NumPy implementations match
            assert abs(torch_result.item() - numpy_result) < 1e-5, \
                f"Threshold {threshold}: PyTorch: {torch_result.item()}, NumPy: {numpy_result}" 