import pytest
import torch
import numpy as np
from data.transforms.vision_3d.pcr_translation import PCRTranslation
from utils.point_cloud_ops import apply_transform


def create_random_point_cloud(num_points=1000, scale=1000000.0):
    """Create a random point cloud with extreme coordinate values."""
    # Generate random points with extreme values
    points = np.random.randn(num_points, 3) * scale
    # Add some offset to make coordinates more extreme
    points += np.array([500000.0, 4800000.0, 100.0])
    return torch.tensor(points, dtype=torch.float32)


def create_random_transform():
    """Create a random 4x4 transformation matrix."""
    # Create a random rotation matrix
    angle = np.random.rand() * 2 * np.pi
    axis = np.random.rand(3)
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    # Create a random translation vector
    t = np.random.rand(3) * 1000.0
    
    # Combine into a 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    return torch.tensor(transform, dtype=torch.float32)


def create_point_cloud_dict(points):
    """Create a point cloud dictionary with the given points."""
    return {
        'pos': points,
        'feat': torch.ones((points.shape[0], 1), dtype=torch.float32),
    }


@pytest.mark.parametrize("num_points", [100, 1000, 5000])
def test_pcr_translation(num_points):
    """Test the PCRTranslation transform with different point cloud sizes."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a random source point cloud with extreme coordinates
    src_points = create_random_point_cloud(num_points)
    src_pc = create_point_cloud_dict(src_points)
    
    # Create a random transformation matrix
    transform = create_random_transform()
    
    # Apply the transformation to create a target point cloud
    tgt_points = apply_transform(src_points, transform)
    tgt_pc = create_point_cloud_dict(tgt_points)
    
    # Create and apply the PCRTranslation transform
    pcr_translation = PCRTranslation()
    new_src_pc, new_tgt_pc, new_transform = pcr_translation(src_pc, tgt_pc, transform)
    
    # 1. Check that only translation happened (no rotation or scaling or non-rigid deformation)
    src_translation = new_src_pc['pos'] - src_pc['pos']
    tgt_translation = new_tgt_pc['pos'] - tgt_pc['pos']
    assert torch.allclose(src_translation[0], src_translation[1:], atol=1e-6)
    assert torch.allclose(tgt_translation[0], tgt_translation[1:], atol=1e-6)
    assert torch.allclose(new_transform[:3, :3], transform[:3, :3], atol=1e-6)
    
    # 2. Check that the translations are consistent
    # The translation applied to src_pc and tgt_pc should be similar to the change in the transform's translation
    # The translation vectors should be similar in magnitude but opposite in direction
    # because the transform's translation is applied in the opposite direction
    transform_translation_change = new_transform[:3, 3] - transform[:3, 3]
    src_translation_vector = src_translation[0]
    tgt_translation_vector = tgt_translation[0]
    assert torch.allclose(src_translation_vector, -transform_translation_change, atol=1e-6)
    assert torch.allclose(tgt_translation_vector, transform_translation_change, atol=1e-6)

    # 3. Check that the mean of the union of the new point clouds is close to zero
    union_points = torch.cat([new_src_pc['pos'], new_tgt_pc['pos']], dim=0)
    mean = union_points.mean(dim=0)
    assert torch.allclose(mean, torch.zeros(3), atol=1e-6)
    
    # 4. Check validity of the output transform matrix
    transformed_src = apply_transform(new_src_pc['pos'], new_transform)
    assert torch.allclose(transformed_src, new_tgt_pc['pos'], atol=1e-6)
