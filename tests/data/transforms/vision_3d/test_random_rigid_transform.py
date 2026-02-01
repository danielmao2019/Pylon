import numpy as np
import pytest
import torch

from data.structures.three_d.camera.rotation.rodrigues import rodrigues_to_matrix
from data.structures.three_d.point_cloud.ops import apply_transform
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.transforms.vision_3d.random_rigid_transform import RandomRigidTransform

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_random_point_cloud(num_points=1000):
    """Create a random point cloud."""
    return torch.randn(size=(num_points, 3), dtype=torch.float32, device=DEVICE)


def create_random_transform():
    """Create a random 4x4 transformation matrix."""
    # Create a random rotation using Rodrigues representation
    angle = np.random.rand() * 2 * np.pi
    axis = np.random.rand(3).astype(np.float32)
    axis = axis / np.linalg.norm(axis)  # Normalize to unit vector

    # Convert to torch tensors
    axis_torch = torch.tensor(axis, dtype=torch.float32, device=DEVICE)
    angle_torch = torch.tensor(angle, dtype=torch.float32, device=DEVICE)

    # Create rotation matrix using rodrigues_to_matrix utility
    R = rodrigues_to_matrix(axis_torch, angle_torch)

    # Create a random translation vector
    t = np.random.rand(3).astype(np.float32) * 10.0
    t_torch = torch.tensor(t, dtype=torch.float32, device=DEVICE)

    # Combine into a 4x4 transformation matrix
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = R
    transform[:3, 3] = t_torch

    return transform.to(DEVICE)


def create_point_cloud(points: torch.Tensor) -> PointCloud:
    """Create a PointCloud with a feature field."""
    return PointCloud(xyz=points, data={'feat': torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)})


def test_random_rigid_transform():
    """Test the RandomRigidTransform by validating the transformed triplet."""
    # 1. Create a random source point cloud
    src_points = create_random_point_cloud(1000)
    src_pc = create_point_cloud(src_points)
    original_src_xyz = src_pc.xyz.clone()
    original_src_feat = src_pc.feat.clone()

    # 2. Create a random transformation matrix
    transform = create_random_transform()

    # 3. Apply the transformation to create a target point cloud
    tgt_points = apply_transform(src_points, transform)
    tgt_pc = create_point_cloud(tgt_points)
    original_tgt_xyz = tgt_pc.xyz.clone()
    original_tgt_feat = tgt_pc.feat.clone()

    # 4. Create and apply the RandomRigidTransform
    random_rigid_transform = RandomRigidTransform(rot_mag=45.0, trans_mag=0.5)
    new_src_pc, new_tgt_pc, new_transform = random_rigid_transform(src_pc, tgt_pc, transform, seed=42)

    # 5. Validate the new triplet
    # Apply the new transform to the new source point cloud
    transformed_src = apply_transform(new_src_pc.xyz, new_transform)

    # Check that the transformed source is (almost) exactly the same as the new target
    assert torch.allclose(transformed_src, new_tgt_pc.xyz, atol=1e-6), \
        f"Transformed source does not match new target. Max difference: {(transformed_src - new_tgt_pc.xyz).abs().max()}"

    # 6. Additional checks
    # Check that the target point cloud is unchanged
    assert torch.allclose(new_tgt_pc.xyz, original_tgt_xyz, atol=1e-6), \
        f"Target point cloud was modified. Max difference: {(new_tgt_pc.xyz - original_tgt_xyz).abs().max()}"

    # Check that the source point cloud was transformed
    assert not torch.allclose(new_src_pc.xyz, original_src_xyz, atol=1e-6), \
        "Source point cloud was not transformed"

    # Check that the feature fields are preserved
    assert torch.allclose(new_src_pc.feat, original_src_feat, atol=1e-6), \
        "Source feature field was modified"
    assert torch.allclose(new_tgt_pc.feat, original_tgt_feat, atol=1e-6), \
        "Target feature field was modified"


def test_random_rigid_transform_deterministic():
    """Test that the RandomRigidTransform produces deterministic results with the same seed."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a random source point cloud
    src_points = create_random_point_cloud(1000)
    src_pc = create_point_cloud(src_points)

    # Create a random transformation matrix
    transform = create_random_transform()

    # Apply the transformation to create a target point cloud
    tgt_points = apply_transform(src_points, transform)
    tgt_pc = create_point_cloud(tgt_points)

    # Create the RandomRigidTransform
    random_rigid_transform = RandomRigidTransform(rot_mag=45.0, trans_mag=0.5)

    # Apply the transform twice with the same seed
    new_src_pc1, new_tgt_pc1, new_transform1 = random_rigid_transform(
        create_point_cloud(src_points.clone()),
        create_point_cloud(tgt_points.clone()),
        transform,
        seed=42,
    )
    new_src_pc2, new_tgt_pc2, new_transform2 = random_rigid_transform(
        create_point_cloud(src_points.clone()),
        create_point_cloud(tgt_points.clone()),
        transform,
        seed=42,
    )

    # Check that the results are identical
    assert torch.allclose(new_src_pc1.xyz, new_src_pc2.xyz, atol=1e-6), \
        "Results are not deterministic with the same seed"
    assert torch.allclose(new_transform1, new_transform2, atol=1e-6), \
        "Transforms are not deterministic with the same seed"
