import pytest
import torch
import numpy as np
from scipy.spatial import KDTree

from metrics.vision_3d import RMSE


def compute_rmse_numpy(transformed, target):
    """Original numpy implementation of RMSE."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(transformed)
    return np.sqrt(np.mean(distances ** 2))


def compute_rmse_with_correspondences_numpy(transformed, target):
    """Original numpy implementation to compute RMSE with correspondences."""
    kdtree = KDTree(target)
    distances, indices = kdtree.query(transformed)
    rmse = np.sqrt(np.mean(distances ** 2))
    return rmse, indices


def compute_rmse_torch(transformed, target):
    """PyTorch implementation of RMSE."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)
    
    # Find nearest neighbor distances
    min_distances, min_indices = torch.min(dist_matrix, dim=1)  # (N,), (N,)
    
    # Compute RMSE
    rmse = torch.sqrt(torch.mean(min_distances ** 2))
    
    return rmse, min_indices


class TestRMSE:
    def test_rmse(self):
        """Test RMSE calculation."""
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
        
        # Compute RMSE using PyTorch implementation
        torch_result, _ = compute_rmse_torch(source_torch, target_torch)
        
        # Compute RMSE using NumPy implementation
        numpy_result = compute_rmse_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
    
    def test_compute_with_correspondences(self):
        """Test RMSE with correspondences."""
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
        
        # Compute RMSE with correspondences using PyTorch implementation
        torch_rmse, torch_indices = compute_rmse_torch(source_torch, target_torch)
        
        # Convert PyTorch tensor to numpy array for comparison
        torch_indices_np = torch_indices.cpu().numpy()
        
        # Compute RMSE with correspondences using NumPy implementation
        numpy_rmse, numpy_indices = compute_rmse_with_correspondences_numpy(source_np, target_np)
        
        # Check that the RMSE values are approximately equal
        assert abs(torch_rmse.item() - numpy_rmse) < 1e-5, f"PyTorch RMSE: {torch_rmse.item()}, NumPy RMSE: {numpy_rmse}"
        
        # Check that the correspondences match
        assert np.array_equal(torch_indices_np, numpy_indices), f"PyTorch indices: {torch_indices_np}, NumPy indices: {numpy_indices}"
    
    def test_with_random_point_clouds(self):
        """Test with randomly generated point clouds."""
        # Generate random point clouds
        np.random.seed(42)
        source_np = np.random.randn(100, 3)
        target_np = np.random.randn(150, 3)
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Compute RMSE using PyTorch implementation
        torch_result, _ = compute_rmse_torch(source_torch, target_torch)
        
        # Compute RMSE using NumPy implementation
        numpy_result = compute_rmse_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"
    
    def test_known_distance(self):
        """Test with known distances to verify correctness."""
        # Create point clouds with known distances
        source_np = np.array([
            [0.0, 0.0, 0.0],  # Distance to nearest target: 0.1732
            [1.0, 0.0, 0.0],  # Distance to nearest target: 0.1732
        ])
        target_np = np.array([
            [0.1, 0.1, 0.1],  # Distance from first source: sqrt(0.01+0.01+0.01) = 0.1732
            [1.1, 0.1, 0.1],  # Distance from second source: sqrt(0.01+0.01+0.01) = 0.1732
        ])
        
        # Expected RMSE: sqrt((0.1732^2 + 0.1732^2)/2) = 0.1732
        expected_rmse = np.sqrt((0.1732**2 + 0.1732**2)/2)
        
        # Convert to PyTorch tensors
        source_torch = torch.tensor(source_np, dtype=torch.float32)
        target_torch = torch.tensor(target_np, dtype=torch.float32)
        
        # Compute RMSE using PyTorch implementation
        torch_result, _ = compute_rmse_torch(source_torch, target_torch)
        
        # Check that the result is approximately equal to expected
        assert abs(torch_result.item() - expected_rmse) < 1e-3, f"PyTorch: {torch_result.item()}, Expected: {expected_rmse}" 