import pytest
import torch
import numpy as np
from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform
from utils.point_cloud_ops import apply_transform


def create_random_point_cloud(num_points=1000):
    """Create a random point cloud."""
    points = torch.randn(num_points, 3, dtype=torch.float32)
    return points


def create_random_transform():
    """Create a random 4x4 transformation matrix."""
    # Create a random rotation matrix
    angle = np.random.rand() * 2 * np.pi
    axis = np.random.rand(3).astype(np.float32)
    axis = axis / np.linalg.norm(axis)

    # Rodrigues rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Create a random translation vector
    t = np.random.rand(3).astype(np.float32) * 10.0

    # Combine into a 4x4 transformation matrix
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = t

    return torch.tensor(transform, dtype=torch.float32)


def create_point_cloud_dict(points):
    """Create a point cloud dictionary with the given points."""
    return {
        'pos': points,
        'feat': torch.ones((points.shape[0], 1), dtype=torch.float32),
    }


def test_random_rigid_transform():
    """Test the RandomRigidTransform by validating the transformed triplet."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Create a random source point cloud
    src_points = create_random_point_cloud(1000)
    src_pc = create_point_cloud_dict(src_points)

    # 2. Create a random transformation matrix
    transform = create_random_transform()

    # 3. Apply the transformation to create a target point cloud
    tgt_points = apply_transform(src_points, transform)
    tgt_pc = create_point_cloud_dict(tgt_points)

    # 4. Create and apply the RandomRigidTransform
    random_rigid_transform = RandomRigidTransform(rot_mag=45.0, trans_mag=0.5)
    new_src_pc, new_tgt_pc, new_transform = random_rigid_transform(src_pc, tgt_pc, transform)

    # 5. Validate the new triplet
    # Apply the new transform to the new source point cloud
    transformed_src = apply_transform(new_src_pc['pos'], new_transform)
    
    # Check that the transformed source is (almost) exactly the same as the new target
    assert torch.allclose(transformed_src, new_tgt_pc['pos'], atol=1e-6), \
        f"Transformed source does not match new target. Max difference: {(transformed_src - new_tgt_pc['pos']).abs().max()}"

    # 6. Additional checks
    # Check that the target point cloud is unchanged
    assert torch.allclose(new_tgt_pc['pos'], tgt_pc['pos'], atol=1e-6), \
        f"Target point cloud was modified. Max difference: {(new_tgt_pc['pos'] - tgt_pc['pos']).abs().max()}"
    
    # Check that the source point cloud was transformed
    assert not torch.allclose(new_src_pc['pos'], src_pc['pos'], atol=1e-6), \
        "Source point cloud was not transformed"
    
    # Check that the feature fields are preserved
    assert torch.allclose(new_src_pc['feat'], src_pc['feat'], atol=1e-6), \
        "Source feature field was modified"
    assert torch.allclose(new_tgt_pc['feat'], tgt_pc['feat'], atol=1e-6), \
        "Target feature field was modified"


def test_random_rigid_transform_deterministic():
    """Test that the RandomRigidTransform produces deterministic results with the same seed."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a random source point cloud
    src_points = create_random_point_cloud(1000)
    src_pc = create_point_cloud_dict(src_points)

    # Create a random transformation matrix
    transform = create_random_transform()

    # Apply the transformation to create a target point cloud
    tgt_points = apply_transform(src_points, transform)
    tgt_pc = create_point_cloud_dict(tgt_points)

    # Create the RandomRigidTransform
    random_rigid_transform = RandomRigidTransform(rot_mag=45.0, trans_mag=0.5)

    # Apply the transform twice with the same seed
    torch.manual_seed(42)
    np.random.seed(42)
    new_src_pc1, new_tgt_pc1, new_transform1 = random_rigid_transform(src_pc, tgt_pc, transform)

    torch.manual_seed(42)
    np.random.seed(42)
    new_src_pc2, new_tgt_pc2, new_transform2 = random_rigid_transform(src_pc, tgt_pc, transform)

    # Check that the results are identical
    assert torch.allclose(new_src_pc1['pos'], new_src_pc2['pos'], atol=1e-6), \
        "Results are not deterministic with the same seed"
    assert torch.allclose(new_transform1, new_transform2, atol=1e-6), \
        "Transforms are not deterministic with the same seed"
