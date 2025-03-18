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


class TestChamferDistance:
    def test_chamfer_distance_bidirectional(self):
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
        metric = ChamferDistance(bidirectional=True)
        torch_result = metric._compute_score(source_torch, target_torch)["chamfer_distance"].item()
        
        # Compute Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result - numpy_result) < 1e-5, f"PyTorch: {torch_result}, NumPy: {numpy_result}"
    
    def test_chamfer_distance_unidirectional(self):
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
        metric = ChamferDistance(bidirectional=False)
        torch_result = metric._compute_score(source_torch, target_torch)["chamfer_distance"].item()
        
        # Compute unidirectional Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_unidirectional_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result - numpy_result) < 1e-5, f"PyTorch: {torch_result}, NumPy: {numpy_result}"
    
    def test_with_random_point_clouds(self):
        # Generate random point clouds of different sizes
        np.random.seed(42)
        source_np = np.random.randn(100, 3)
        target_np = np.random.randn(150, 3)
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Compute Chamfer distance using PyTorch implementation
        metric = ChamferDistance(bidirectional=True)
        torch_result = metric._compute_score(source_torch, target_torch)["chamfer_distance"].item()
        
        # Compute Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result - numpy_result) < 1e-5, f"PyTorch: {torch_result}, NumPy: {numpy_result}"
