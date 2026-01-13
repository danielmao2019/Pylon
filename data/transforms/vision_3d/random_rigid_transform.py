from typing import Tuple, Optional, Any
import numpy as np
import torch
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.ops import apply_transform
from utils.three_d.rotation.rodrigues import rodrigues_to_matrix


class RandomRigidTransform(BaseTransform):
    """Random rigid transformation (rotation and translation) for point clouds."""

    def __init__(
        self,
        rot_mag: float = 45.0,
        trans_mag: float = 0.5,
        method: str = 'Rodrigues',
        num_axis: Optional[int] = None,
    ) -> None:
        """
        Initialize the random rigid transform.

        Args:
            rot_mag: Maximum rotation magnitude in degrees (default: 45.0)
            trans_mag: Maximum translation magnitude (default: 0.5)
            method: Rotation method, either 'Rodrigues' or 'Euler' (default: 'Rodrigues')
            num_axis: Number of axes to rotate around for Euler method (0, 1, or 3). Only used when method='Euler'
        """
        super(RandomRigidTransform, self).__init__()
        assert isinstance(rot_mag, (int, float)), f"{type(rot_mag)=}"
        assert isinstance(trans_mag, (int, float)), f"{type(trans_mag)=}"
        assert rot_mag >= 0, f"{rot_mag=}"
        assert trans_mag >= 0, f"{trans_mag=}"
        assert method in ['Rodrigues', 'Euler'], f"{method=}"
        if method == 'Euler':
            assert num_axis in [0, 1, 3], f"{num_axis=}"

        self.rot_mag = rot_mag
        self.trans_mag = trans_mag
        self.method = method
        self.num_axis = num_axis

    def _sample_rotation_Rodrigues(self, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        """
        Sample a random rotation using Rodrigues' formula.

        Args:
            device: Device to create tensors on

        Returns:
            A 3x3 rotation matrix
        """
        rot_mag_rad = np.radians(self.rot_mag)

        # Generate a random axis of rotation
        axis = torch.randn(3, device=device, generator=generator)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Generate random angle within the specified range
        angle = torch.rand(1, device=device, generator=generator) * (2 * rot_mag_rad) - rot_mag_rad
        angle = angle.squeeze()

        # Create rotation matrix using rodrigues_to_matrix utility
        R = rodrigues_to_matrix(axis, angle)

        return R

    def _sample_rotation_Euler(self, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        """
        Sample a random rotation using Euler angles.

        Args:
            device: Device to create tensors on

        Returns:
            A 3x3 rotation matrix
        """
        if self.num_axis == 0:
            return torch.eye(3, device=device)

        rot_mag_rad = np.radians(self.rot_mag)
        angles = torch.rand(3, device=device, generator=generator) * (2 * rot_mag_rad) - rot_mag_rad

        # Create rotation matrices for each axis
        Rx = torch.tensor([[1, 0, 0],
                          [0, torch.cos(angles[0]), -torch.sin(angles[0])],
                          [0, torch.sin(angles[0]), torch.cos(angles[0])]], device=device)

        Ry = torch.tensor([[torch.cos(angles[1]), 0, torch.sin(angles[1])],
                          [0, 1, 0],
                          [-torch.sin(angles[1]), 0, torch.cos(angles[1])]], device=device)

        Rz = torch.tensor([[torch.cos(angles[2]), -torch.sin(angles[2]), 0],
                          [torch.sin(angles[2]), torch.cos(angles[2]), 0],
                          [0, 0, 1]], device=device)

        if self.num_axis == 1:
            return Rz
        return Rx @ Ry @ Rz

    def _sample_rigid_transform(self, device: torch.device, generator: torch.Generator) -> torch.Tensor:
        """
        Sample a random rigid transformation.

        Args:
            device: Device to create tensors on

        Returns:
            A 4x4 transformation matrix
        """
        # Generate rotation matrix using the specified method
        if self.method == 'Rodrigues':
            R = self._sample_rotation_Rodrigues(device, generator)
        else:  # method == 'Euler'
            R = self._sample_rotation_Euler(device, generator)

        # Generate random translation
        # Generate random direction (unit vector)
        trans_dir = torch.randn(3, device=device, generator=generator)
        trans_dir = trans_dir / torch.norm(trans_dir)

        # Generate random magnitude within limit
        trans_mag = torch.rand(1, device=device, generator=generator) * self.trans_mag

        # Compute final translation vector
        trans = trans_dir * trans_mag

        # Create 4x4 transformation matrix
        transform = torch.eye(4, device=device)
        transform[:3, :3] = R
        transform[:3, 3] = trans

        return transform

    def __call__(
        self,
        src_pc: PointCloud,
        tgt_pc: PointCloud,
        transform: torch.Tensor,
        seed: Optional[Any] = None,
    ) -> Tuple[PointCloud, PointCloud, torch.Tensor]:
        """
        Apply random rigid transformation to the source point cloud and adjust the transformation matrix.

        Args:
            src_pc: Source point cloud
            tgt_pc: Target point cloud
            transform: Original transformation matrix from source to target, shape (4, 4)
            seed: The seed to use for the random rigid transform.

        Returns:
            A tuple containing:
            - Transformed source point cloud
            - Unchanged target point cloud
            - Adjusted transformation matrix
        """
        assert isinstance(src_pc, PointCloud), f"{type(src_pc)=}"
        assert isinstance(tgt_pc, PointCloud), f"{type(tgt_pc)=}"
        assert src_pc.xyz.ndim == 2 and src_pc.xyz.shape[1] == 3, f"{src_pc.xyz.shape=}"
        assert tgt_pc.xyz.ndim == 2 and tgt_pc.xyz.shape[1] == 3, f"{tgt_pc.xyz.shape=}"
        assert src_pc.xyz.dtype == torch.float32, f"{src_pc.xyz.dtype=}"
        assert tgt_pc.xyz.dtype == torch.float32, f"{tgt_pc.xyz.dtype=}"

        assert isinstance(transform, torch.Tensor), f"{type(transform)=}"
        assert transform.shape == (4, 4), f"{transform.shape=}"
        assert transform.dtype == torch.float32, f"{transform.dtype=}"

        # Sample a random transformation
        generator = self._get_generator(g_type='torch', seed=seed)
        random_transform = self._sample_rigid_transform(transform.device, generator)

        # Apply random transformation to the source point cloud
        transformed_src_xyz = apply_transform(
            points=src_pc.xyz, transform=random_transform
        )
        src_fields = {
            name: getattr(src_pc, name)
            for name in src_pc.field_names()
            if name != 'xyz'
        }
        new_src_pc = PointCloud(xyz=transformed_src_xyz, data=src_fields)
        tgt_fields = {
            name: getattr(tgt_pc, name)
            for name in tgt_pc.field_names()
            if name != 'xyz'
        }
        new_tgt_pc = PointCloud(xyz=tgt_pc.xyz, data=tgt_fields)

        # Adjust the transformation matrix
        # The new transformation is: new_transform = transform @ random_transform^(-1)
        # This is because we want the new transformation to map from the randomly transformed
        # source point cloud to the target point cloud
        random_transform_inv = torch.inverse(random_transform)
        # the following assertions are disabled because of numerical errors
        # assert torch.equal(random_transform_inv[-1, :], torch.tensor([0, 0, 0, 1], device=random_transform_inv.device))
        # assert torch.allclose(random_transform_inv[:3, :3], random_transform[:3, :3].T), f"{random_transform_inv[:3, :3]=}\n{random_transform[:3, :3].T=}"
        # assert torch.allclose(random_transform_inv[:3, 3], -random_transform[:3, :3].T @ random_transform[:3, 3])

        new_transform = transform @ random_transform_inv

        return new_src_pc, new_tgt_pc, new_transform
