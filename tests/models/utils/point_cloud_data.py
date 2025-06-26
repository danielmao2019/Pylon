"""Dummy data generators for point cloud model testing."""
from typing import Dict
import torch


def generate_point_cloud_data(
    batch_size: int = 2,
    num_points: int = 1024,
    feature_dim: int = 32,
    device: str = 'cpu'
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Generate dummy data for point cloud registration models.
    
    Note: Point clouds follow batched format (batch_size, num_points, 3) for compatibility
    with existing registration models. For segmentation tasks, use flattened format.
    
    Args:
        batch_size: Number of samples in batch
        num_points: Number of points per cloud
        feature_dim: Feature dimension
        device: Device to place tensors on
        
    Returns:
        Dictionary with 'src_pc' and 'tgt_pc' containing 'pos' and 'feat'
    """
    return {
        'src_pc': {
            'pos': torch.randn(batch_size, num_points, 3, dtype=torch.float32, device=device),
            'feat': torch.randn(batch_size, num_points, feature_dim, dtype=torch.float32, device=device)
        },
        'tgt_pc': {
            'pos': torch.randn(batch_size, num_points, 3, dtype=torch.float32, device=device),
            'feat': torch.randn(batch_size, num_points, feature_dim, dtype=torch.float32, device=device)
        }
    }


def generate_point_cloud_segmentation_data(
    num_points: int = 1024,
    feature_dim: int = 32,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """Generate dummy data for point cloud segmentation (flattened format).
    
    Args:
        num_points: Total number of points (concatenated across batches)
        feature_dim: Feature dimension
        device: Device to place tensors on
        
    Returns:
        Dictionary with flattened point cloud data
    """
    return {
        'pos': torch.randn(num_points, 3, dtype=torch.float32, device=device),
        'feat': torch.randn(num_points, feature_dim, dtype=torch.float32, device=device)
    }


def generate_point_cloud_segmentation_labels(
    num_points: int = 1024,
    num_classes: int = 10,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate dummy labels for point cloud segmentation.
    
    Args:
        num_points: Total number of points
        num_classes: Number of classes
        device: Device to place tensors on
        
    Returns:
        Point cloud segmentation labels (N,) int64
    """
    return torch.randint(0, num_classes, (num_points,), dtype=torch.int64, device=device)


def generate_transformation_matrix(
    batch_size: int = 2,
    device: str = 'cpu'
) -> torch.Tensor:
    """Generate dummy transformation matrices for point cloud registration.
    
    Args:
        batch_size: Number of samples in batch
        device: Device to place tensors on
        
    Returns:
        4x4 transformation matrices (N, 4, 4) float32
    """
    # Generate random rotation matrices and translation vectors
    angles = torch.randn(batch_size, 3, dtype=torch.float32, device=device) * 0.1  # Small rotations
    translations = torch.randn(batch_size, 3, dtype=torch.float32, device=device) * 0.5
    
    # Create transformation matrices
    T = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    T[:, :3, 3] = translations
    
    return T


def create_minimal_point_cloud_input(device: str = 'cpu') -> Dict[str, Dict[str, torch.Tensor]]:
    """Create minimal valid input for testing edge cases.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        Minimal valid input data
    """
    return generate_point_cloud_data(
        batch_size=1, num_points=64, device=device
    )


def create_large_point_cloud_input(device: str = 'cpu') -> Dict[str, Dict[str, torch.Tensor]]:
    """Create larger input for memory testing.
    
    Args:
        device: Device to place tensors on
        
    Returns:
        Large input data for memory testing
    """
    return generate_point_cloud_data(
        batch_size=4, num_points=4096, device=device
    )