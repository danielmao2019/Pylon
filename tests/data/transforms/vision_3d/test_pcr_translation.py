import numpy as np
import pytest
import torch

from data.structures.three_d.camera.rotation.rodrigues import rodrigues_to_matrix
from data.structures.three_d.point_cloud.ops import apply_transform
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.transforms.vision_3d.pcr_translation import PCRTranslation


def create_random_point_cloud(num_points=1000, scale=1000):
    """Create a random point cloud with extreme coordinate values."""
    # Generate random points with extreme values
    points = np.random.randn(num_points, 3).astype(np.float32) * scale
    # Add some offset to make coordinates more extreme
    points += np.array([5.0, 48.0, 1.0], dtype=np.float32) * scale
    return torch.tensor(points, dtype=torch.float32)


def create_random_transform():
    """Create a random 4x4 transformation matrix."""
    # Create a random rotation using Rodrigues representation
    angle = np.random.rand() * 2 * np.pi
    axis = np.random.rand(3).astype(np.float32)
    axis = axis / np.linalg.norm(axis)  # Normalize to unit vector

    # Convert to torch tensors
    axis_torch = torch.tensor(axis, dtype=torch.float32)
    angle_torch = torch.tensor(angle, dtype=torch.float32)

    # Create rotation matrix using rodrigues_to_matrix utility
    R = rodrigues_to_matrix(axis_torch, angle_torch)

    # Create a random translation vector
    t = np.random.rand(3).astype(np.float32) * 1000.0
    t_torch = torch.tensor(t, dtype=torch.float32)

    # Combine into a 4x4 transformation matrix
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = R
    transform[:3, 3] = t_torch

    return transform


def create_point_cloud(points: torch.Tensor) -> PointCloud:
    """Create a PointCloud with a feature field."""
    return PointCloud(xyz=points, data={'feat': torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)})


@pytest.mark.parametrize("num_points", [100, 1000, 5000])
def test_pcr_translation(num_points):
    """Test the PCRTranslation transform with different point cloud sizes."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create a random source point cloud with extreme coordinates
    src_points = create_random_point_cloud(num_points)
    src_pc = create_point_cloud(src_points)

    # Create a random transformation matrix
    transform = create_random_transform()

    # Apply the transformation to create a target point cloud
    tgt_points = apply_transform(src_points, transform)
    tgt_pc = create_point_cloud(tgt_points)

    # Create and apply the PCRTranslation transform
    pcr_translation = PCRTranslation()
    new_src_pc, new_tgt_pc, new_transform = pcr_translation(src_pc, tgt_pc, transform)

    # 1. Check that only translation happened (no rotation or scaling or non-rigid deformation)
    src_translation = new_src_pc.xyz - src_points
    assert torch.allclose(src_translation[0], src_translation[1:], atol=1e-6), \
        f"Source translation is not uniform across points. First point translation: {src_translation[0]}, others: {src_translation[1:]}"
    tgt_translation = new_tgt_pc.xyz - tgt_points
    assert torch.allclose(tgt_translation[0], tgt_translation[1:], atol=1e-6), \
        f"Target translation is not uniform across points. First point translation: {tgt_translation[0]}, others: {tgt_translation[1:]}"

    # 2. Check that the translations are consistent
    assert torch.allclose(src_translation[0], tgt_translation[0], atol=1e-6), \
        f"Source and target translations are not consistent. Source: {src_translation[0]}, Target: {tgt_translation[0]}"

    # 3. Check that the mean of the union of the new point clouds is close to zero
    union_points = torch.cat([new_src_pc.xyz, new_tgt_pc.xyz], dim=0)
    mean = union_points.mean(dim=0)
    assert torch.allclose(mean, torch.zeros(3, dtype=torch.float32), atol=1e-2), \
        f"Union mean is not close to zero. Mean: {mean}"

    # 4. Check validity of the output transform matrix
    transformed_src = apply_transform(new_src_pc.xyz, new_transform)
    assert torch.allclose(transformed_src, new_tgt_pc.xyz, atol=1e-6), \
        f"Transform is not valid. Max difference: {(transformed_src - new_tgt_pc.xyz).abs().max()}"
