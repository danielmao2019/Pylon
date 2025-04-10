import pytest
import torch
import numpy as np
from scipy.spatial import KDTree

from metrics.vision_3d import ChamferDistance


def compute_chamfer_distance_numpy(transformed, target):
    """Original numpy implementation of chamfer distance."""
    kdtree_source = KDTree(transformed)
    kdtree_target = KDTree(target)
    
    # Distance from transformed to target
    dist_source_to_target, _ = kdtree_target.query(transformed)
    
    # Distance from target to transformed
    dist_target_to_source, _ = kdtree_source.query(target)
    
    # The Chamfer Distance is the sum of means
    return np.mean(dist_source_to_target) + np.mean(dist_target_to_source)


def compute_chamfer_distance_unidirectional_numpy(source, target):
    """Original numpy implementation of unidirectional chamfer distance."""
    kdtree_target = KDTree(target)
    dist_source_to_target, _ = kdtree_target.query(source)
    return np.mean(dist_source_to_target)


def compute_chamfer_distance_torch(transformed, target, bidirectional=True):
    """PyTorch implementation of chamfer distance."""
    # Compute nearest neighbor distances from transformed to target
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)
    
    # Find nearest neighbor distances
    min_dist_transformed_to_target = torch.min(dist_matrix, dim=1)[0]  # (N,)
    
    if bidirectional:
        min_dist_target_to_transformed = torch.min(dist_matrix, dim=0)[0]  # (M,)
        return torch.mean(min_dist_transformed_to_target) + torch.mean(min_dist_target_to_transformed)
    else:
        return torch.mean(min_dist_transformed_to_target)


class TestChamferDistance:
    def test_chamfer_distance_bidirectional(self):
        """Test bidirectional Chamfer distance implementation."""
        # Create sample point clouds
        source_np = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        target_np = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.1, 1.1, 0.1],
            [0.5, 0.5, 0.5]  # Additional point
        ])
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Compute Chamfer distance using PyTorch implementation
        torch_result = compute_chamfer_distance_torch(source_torch, target_torch, bidirectional=True)
        
        # Compute Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
    
    def test_chamfer_distance_unidirectional(self):
        """Test unidirectional Chamfer distance implementation."""
        # Create sample point clouds
        source_np = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        target_np = np.array([
            [0.1, 0.1, 0.1],
            [1.1, 0.1, 0.1],
            [0.1, 1.1, 0.1],
            [1.1, 1.1, 0.1],
            [0.5, 0.5, 0.5]  # Additional point
        ])
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Compute unidirectional Chamfer distance using PyTorch implementation
        torch_result = compute_chamfer_distance_torch(source_torch, target_torch, bidirectional=False)
        
        # Compute unidirectional Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_unidirectional_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
    
    def test_with_random_point_clouds(self):
        """Test with randomly generated point clouds."""
        # Generate random point clouds of different sizes
        np.random.seed(42)
        source_np = np.random.randn(100, 3)
        target_np = np.random.randn(150, 3)
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Compute Chamfer distance using PyTorch implementation
        torch_result = compute_chamfer_distance_torch(source_torch, target_torch, bidirectional=True)
        
        # Compute Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
