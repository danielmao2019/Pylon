import pytest
import torch
import numpy as np
from scipy.spatial import KDTree

# Import all individual test modules
from tests.metrics.vision_3d.test_chamfer_distance import TestChamferDistance
from tests.metrics.vision_3d.test_rmse import TestRMSE
from tests.metrics.vision_3d.test_mae import TestMAE
from tests.metrics.vision_3d.test_inlier_ratio import TestInlierRatio
from tests.metrics.vision_3d.test_registration_recall import TestRegistrationRecall
from tests.metrics.vision_3d.test_precision_recall import TestCorrespondencePrecisionRecall
from tests.metrics.vision_3d.test_point_cloud_confusion_matrix import TestPointCloudConfusionMatrix


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


def compute_chamfer_distance_torch(transformed, target):
    """PyTorch implementation of chamfer distance."""
    # Compute nearest neighbor distances from transformed to target
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)
    
    # Find nearest neighbor distances
    min_dist_transformed_to_target = torch.min(dist_matrix, dim=1)[0]  # (N,)
    min_dist_target_to_transformed = torch.min(dist_matrix, dim=0)[0]  # (M,)
    
    # The Chamfer Distance is the sum of means
    return torch.mean(min_dist_transformed_to_target) + torch.mean(min_dist_target_to_transformed)


def compute_rmse_numpy(transformed, target):
    """Original numpy implementation of RMSE."""
    kdtree = KDTree(target)
    distances, _ = kdtree.query(transformed)
    return np.sqrt(np.mean(distances ** 2))


def compute_rmse_torch(transformed, target):
    """PyTorch implementation of RMSE."""
    # Compute nearest neighbor distances
    transformed_expanded = transformed.unsqueeze(1)  # (N, 1, 3)
    target_expanded = target.unsqueeze(0)  # (1, M, 3)
    dist_matrix = torch.sqrt(((transformed_expanded - target_expanded) ** 2).sum(dim=2))  # (N, M)
    
    # Find nearest neighbor distances
    min_distances = torch.min(dist_matrix, dim=1)[0]  # (N,)
    
    # Compute RMSE
    return torch.sqrt(torch.mean(min_distances ** 2))


class TestEquivalence:
    def test_chamfer_distance(self):
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
        torch_result = compute_chamfer_distance_torch(source_torch, target_torch).item()
        
        # Compute Chamfer distance using NumPy implementation
        numpy_result = compute_chamfer_distance_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result - numpy_result) < 1e-5, f"PyTorch: {torch_result}, NumPy: {numpy_result}"
    
    def test_rmse(self):
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
        torch_result = compute_rmse_torch(source_torch, target_torch).item()
        
        # Compute RMSE using NumPy implementation
        numpy_result = compute_rmse_numpy(source_np, target_np)
        
        # Check that the results are approximately equal
        assert abs(torch_result - numpy_result) < 1e-5, f"PyTorch: {torch_result}, NumPy: {numpy_result}" 