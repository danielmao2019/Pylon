"""Tests for point cloud operations."""
import numpy as np
import torch
import pytest
from utils.point_cloud_ops import apply_transform


@pytest.fixture
def random_point_cloud():
    """Fixture to generate a random point cloud."""
    def _generate(num_points=None, use_numpy=False):
        if num_points is None:
            num_points = np.random.randint(100, 1000)
        if use_numpy:
            return np.random.rand(num_points, 3).astype(np.float32)
        else:
            return torch.rand(num_points, 3, dtype=torch.float32)
    return _generate


@pytest.fixture
def random_transform():
    """Fixture to generate a random 4x4 transformation matrix."""
    def _generate(use_numpy=False):
        # Create a random rotation matrix using Rodrigues formula
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.rand(3).astype(np.float32)
        axis = axis / np.linalg.norm(axis)
        
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], dtype=np.float32)
        R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        # Create a random translation vector
        t = np.random.rand(3).astype(np.float32) * 10
        
        # Create the 4x4 transform matrix
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        if not use_numpy:
            transform = torch.tensor(transform, dtype=torch.float32)
            
        return transform
    return _generate


def original_apply_transform(points, transform):
    """
    Original implementation of apply_transform from display_pcr.py.
    Uses rotation matrix and translation vector approach.
    
    Args:
        points: torch.Tensor of shape (N, 3) - point cloud coordinates
        transform: Union[List[List[Union[int, float]]], numpy.ndarray] - transformation matrix

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


@pytest.mark.parametrize("transform_type", ["numpy", "list", "torch"])
@pytest.mark.parametrize("points_type", ["numpy", "torch"])
def test_apply_transform_output_shape(random_point_cloud, random_transform, transform_type, points_type):
    """Test that apply_transform maintains the correct output shape and type."""
    use_numpy = points_type == "numpy"
    points = random_point_cloud(100, use_numpy=use_numpy)
    transform = random_transform(use_numpy=transform_type == "numpy")
    
    if transform_type == "list":
        transform = transform.tolist() if not use_numpy else transform.tolist()
    
    result = apply_transform(points, transform)
    
    # Check shape
    assert result.shape == points.shape, f"Expected shape {points.shape}, got {result.shape}"
    
    # Check type
    if use_numpy:
        assert isinstance(result, np.ndarray), "Expected numpy array output for numpy input"
    else:
        assert isinstance(result, torch.Tensor), "Expected torch tensor output for torch input"


@pytest.mark.parametrize("transform_type", ["numpy", "list", "torch"])
@pytest.mark.parametrize("points_type", ["numpy", "torch"])
def test_apply_transform_equivalence(random_point_cloud, random_transform, transform_type, points_type):
    """Test that apply_transform produces equivalent results to the original implementation."""
    use_numpy = points_type == "numpy"
    points = random_point_cloud(100, use_numpy=use_numpy)
    transform = random_transform(use_numpy=transform_type == "numpy")
    
    if transform_type == "list":
        transform_input = transform.tolist() if not use_numpy else transform.tolist()
    else:
        transform_input = transform
    
    result_new = apply_transform(points, transform_input)
    
    # For comparison with original_apply_transform, convert to torch
    if use_numpy:
        points_torch = torch.tensor(points, dtype=torch.float32)
        transform_torch = torch.tensor(transform, dtype=torch.float32) if isinstance(transform, np.ndarray) else transform
        result_original = original_apply_transform(points_torch, transform_torch)
        result_new_torch = torch.tensor(result_new, dtype=torch.float32)
        assert torch.allclose(result_new_torch, result_original, rtol=1e-5, atol=1e-5), \
            f"Results differ for {points_type} points and {transform_type} transform"
    else:
        result_original = original_apply_transform(points, transform)
        assert torch.allclose(result_new, result_original, rtol=1e-5, atol=1e-5), \
            f"Results differ for {points_type} points and {transform_type} transform"


@pytest.mark.parametrize("points_type", ["numpy", "torch"])
def test_apply_transform_identity(random_point_cloud, points_type):
    """Test that applying identity transform returns the original points."""
    use_numpy = points_type == "numpy"
    points = random_point_cloud(100, use_numpy=use_numpy)
    
    if use_numpy:
        identity = np.eye(4, dtype=np.float32)
    else:
        identity = torch.eye(4, dtype=torch.float32)
    
    result = apply_transform(points, identity)
    
    if use_numpy:
        assert np.allclose(result, points, rtol=1e-5, atol=1e-5), \
            "Identity transform should return original points"
    else:
        assert torch.allclose(result, points, rtol=1e-5, atol=1e-5), \
            "Identity transform should return original points"


@pytest.mark.parametrize("points_type", ["numpy", "torch"])
def test_apply_transform_invalid_shape(random_point_cloud, points_type):
    """Test that apply_transform raises appropriate error for invalid transform shape."""
    use_numpy = points_type == "numpy"
    points = random_point_cloud(100, use_numpy=use_numpy)
    
    if use_numpy:
        invalid_transform = np.eye(3, dtype=np.float32)  # 3x3 instead of 4x4
    else:
        invalid_transform = torch.eye(3, dtype=torch.float32)  # 3x3 instead of 4x4
    
    with pytest.raises(AssertionError, match="Transform must be a 4x4 matrix"):
        apply_transform(points, invalid_transform)


@pytest.mark.parametrize("num_points", [0, 1, 1000])
@pytest.mark.parametrize("points_type", ["numpy", "torch"])
def test_apply_transform_edge_cases(random_transform, num_points, points_type):
    """Test apply_transform with edge cases of point cloud sizes."""
    use_numpy = points_type == "numpy"
    
    if use_numpy:
        points = np.random.rand(num_points, 3).astype(np.float32)
    else:
        points = torch.rand(num_points, 3, dtype=torch.float32)
    
    transform = random_transform(use_numpy=use_numpy)
    
    result = apply_transform(points, transform)
    assert result.shape == points.shape, f"Expected shape {points.shape}, got {result.shape}"
