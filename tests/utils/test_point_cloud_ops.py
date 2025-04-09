"""Tests for point cloud operations."""
import numpy as np
import torch
import pytest
from utils.point_cloud_ops import apply_transform


@pytest.fixture
def random_point_cloud():
    """Fixture to generate a random point cloud."""
    def _generate(num_points=None):
        if num_points is None:
            num_points = np.random.randint(100, 1000)
        return torch.rand(num_points, 3)
    return _generate


@pytest.fixture
def random_transform():
    """Fixture to generate a random 4x4 transformation matrix."""
    def _generate():
        # Create a random rotation matrix using Rodrigues formula
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.rand(3)
        axis = axis / np.linalg.norm(axis)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # Create a random translation vector
        t = np.random.rand(3) * 10
        
        # Create the 4x4 transform matrix
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        return transform
    return _generate


def original_apply_transform(points, transform):
    """
    Original implementation of apply_transform from display_pcr.py.
    
    Args:
        points: torch.Tensor of shape (N, 3) - point cloud coordinates
        transform: Union[List[List[Union[int, float]]], numpy.ndarray, torch.Tensor] - transformation matrix

    Returns:
        torch.Tensor of shape (N, 3) - transformed point cloud coordinates
    """
    # Convert transform to torch.Tensor if it's not already
    if isinstance(transform, list):
        transform = torch.tensor(transform, dtype=torch.float32)
    elif isinstance(transform, np.ndarray):
        transform = torch.tensor(transform, dtype=torch.float32)

    # Ensure transform is a 4x4 matrix
    assert transform.shape == (4, 4), f"Transform must be a 4x4 matrix, got {transform.shape}"

    # Extract rotation and translation
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    # Apply transformation: R * points + t
    transformed_points = torch.matmul(points, rotation.t()) + translation

    return transformed_points


@pytest.mark.parametrize("transform_type", ["numpy", "torch", "list"])
def test_apply_transform_output_shape(random_point_cloud, random_transform, transform_type):
    """Test that apply_transform maintains the correct output shape."""
    points = random_point_cloud(100)
    transform = random_transform()
    
    if transform_type == "torch":
        transform = torch.tensor(transform, dtype=torch.float32)
    elif transform_type == "list":
        transform = transform.tolist()
    
    result = apply_transform(points, transform)
    assert result.shape == points.shape, f"Expected shape {points.shape}, got {result.shape}"


@pytest.mark.parametrize("transform_type", ["numpy", "torch", "list"])
def test_apply_transform_equivalence(random_point_cloud, random_transform, transform_type):
    """Test that apply_transform produces equivalent results to the original implementation."""
    points = random_point_cloud(100)
    transform = random_transform()
    
    if transform_type == "torch":
        transform_input = torch.tensor(transform, dtype=torch.float32)
    elif transform_type == "list":
        transform_input = transform.tolist()
    else:
        transform_input = transform
    
    result_new = apply_transform(points, transform_input)
    result_original = original_apply_transform(points, transform)
    
    assert torch.allclose(result_new, result_original, rtol=1e-5, atol=1e-5), \
        f"Results differ for {transform_type} transform"


def test_apply_transform_identity(random_point_cloud):
    """Test that applying identity transform returns the original points."""
    points = random_point_cloud(100)
    identity = np.eye(4)
    
    result = apply_transform(points, identity)
    assert torch.allclose(result, points, rtol=1e-5, atol=1e-5), \
        "Identity transform should return original points"


def test_apply_transform_invalid_shape(random_point_cloud):
    """Test that apply_transform raises appropriate error for invalid transform shape."""
    points = random_point_cloud(100)
    invalid_transform = np.eye(3)  # 3x3 instead of 4x4
    
    with pytest.raises(AssertionError, match="Transform must be a 4x4 matrix"):
        apply_transform(points, invalid_transform)


@pytest.mark.parametrize("num_points", [0, 1, 1000])
def test_apply_transform_edge_cases(random_transform, num_points):
    """Test apply_transform with edge cases of point cloud sizes."""
    points = torch.rand(num_points, 3)
    transform = random_transform()
    
    result = apply_transform(points, transform)
    assert result.shape == points.shape, f"Expected shape {points.shape}, got {result.shape}"
