import pytest
import torch
import numpy as np
from scipy.spatial import KDTree


def compute_mae_numpy(transformed, target):
    """Original numpy implementation of MAE."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(transformed)
    return np.mean(np.abs(distances))


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


def test_mae():
    """Test MAE calculation."""
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
    
    # Compute MAE using PyTorch implementation
    torch_result = compute_mae_torch(source_torch, target_torch)
    
    # Compute MAE using NumPy implementation
    numpy_result = compute_mae_numpy(source_np, target_np)
    
    # Check that the results are approximately equal
    assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"


def test_with_random_point_clouds():
    """Test with randomly generated point clouds."""
    # Generate random point clouds
    np.random.seed(42)
    source_np = np.random.randn(100, 3)
    target_np = np.random.randn(150, 3)
    
    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    
    # Compute MAE using PyTorch implementation
    torch_result = compute_mae_torch(source_torch, target_torch)
    
    # Compute MAE using NumPy implementation
    numpy_result = compute_mae_numpy(source_np, target_np)
    
    # Check that the results are approximately equal
    assert abs(torch_result.item() - numpy_result) < 1e-5, f"PyTorch: {torch_result.item()}, NumPy: {numpy_result}"


def test_known_distance():
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
    
    # Expected MAE: (0.1732 + 0.1732)/2 = 0.1732
    expected_mae = 0.1732
    
    # Convert to PyTorch tensors
    source_torch = torch.tensor(source_np, dtype=torch.float32)
    target_torch = torch.tensor(target_np, dtype=torch.float32)
    
    # Compute MAE using PyTorch implementation
    torch_result = compute_mae_torch(source_torch, target_torch)
    
    # Check that the result is approximately equal to expected
    assert abs(torch_result.item() - expected_mae) < 1e-3, f"PyTorch: {torch_result.item()}, Expected: {expected_mae}"
