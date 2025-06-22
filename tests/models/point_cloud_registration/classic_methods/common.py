"""Common utilities for classic point cloud registration method tests."""
import torch
from typing import Dict
from tests.models.utils.point_cloud_data import generate_point_cloud_data


def get_dummy_input(device: str = 'cpu', batch_size: int = 2, num_points: int = 512) -> Dict[str, Dict[str, torch.Tensor]]:
    """Get dummy input for point cloud registration."""
    return generate_point_cloud_data(
        batch_size=batch_size, num_points=num_points, feature_dim=32, device=device
    )


def validate_transformation_output(output: torch.Tensor, batch_size: int):
    """Validate transformation matrix output."""
    assert isinstance(output, torch.Tensor)
    assert output.shape == (batch_size, 4, 4)
    
    # Check if transformation matrices are valid
    for b in range(batch_size):
        T = output[b]
        # Bottom row should be [0, 0, 0, 1]
        expected_bottom = torch.tensor([0., 0., 0., 1.], device=T.device)
        bottom_row = T[3, :]
        # Allow some numerical tolerance
        assert torch.allclose(bottom_row, expected_bottom, atol=1e-6)
        
        # Rotation part should be orthogonal (R @ R.T ≈ I)
        R = T[:3, :3]
        identity = torch.eye(3, device=R.device)
        assert torch.allclose(R @ R.T, identity, atol=1e-5)
        
        # Determinant should be 1 (proper rotation)
        det = torch.det(R)
        assert torch.allclose(det, torch.tensor(1.0, device=det.device), atol=1e-5)
