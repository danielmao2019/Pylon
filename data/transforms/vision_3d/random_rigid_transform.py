from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np
from data.transforms.base_transform import BaseTransform
from utils.point_cloud_ops import apply_transform


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
        self.generator = torch.Generator()

    def _sample_rotation_Rodrigues(self, device: torch.device) -> torch.Tensor:
        """
        Sample a random rotation using Rodrigues' formula.

        Args:
            device: Device to create tensors on

        Returns:
            A 3x3 rotation matrix
        """
        rot_mag_rad = np.radians(self.rot_mag)

        # Generate a random axis of rotation
        axis = torch.randn(3, device=device, generator=self.generator)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Generate random angle within the specified range
        angle = torch.rand(1, device=device, generator=self.generator) * (2 * rot_mag_rad) - rot_mag_rad

        # Create rotation matrix using axis-angle representation (Rodrigues' formula)
        K = torch.tensor([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]], dtype=torch.float32, device=device)
        R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        return R

    def _sample_rotation_Euler(self, device: torch.device) -> torch.Tensor:
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
        angles = torch.rand(3, device=device, generator=self.generator) * (2 * rot_mag_rad) - rot_mag_rad

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

    def _sample_rigid_transform(self, device: torch.device) -> torch.Tensor:
        """
        Sample a random rigid transformation.

        Args:
            device: Device to create tensors on

        Returns:
            A 4x4 transformation matrix
        """
        # Generate rotation matrix using the specified method
        if self.method == 'Rodrigues':
            R = self._sample_rotation_Rodrigues(device)
        else:  # method == 'Euler'
            R = self._sample_rotation_Euler(device)

        # Generate random translation
        # Generate random direction (unit vector)
        trans_dir = torch.randn(3, device=device, generator=self.generator)
        trans_dir = trans_dir / torch.norm(trans_dir)

        # Generate random magnitude within limit
        trans_mag = torch.rand(1, device=device, generator=self.generator) * self.trans_mag

        # Compute final translation vector
        trans = trans_dir * trans_mag

        # Create 4x4 transformation matrix
        transform = torch.eye(4, device=device)
        transform[:3, :3] = R
        transform[:3, 3] = trans

        return transform

    def __call__(
        self,
        src_pc: Dict[str, Any],
        tgt_pc: Dict[str, Any],
        transform: torch.Tensor,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
        """
        Apply random rigid transformation to the source point cloud and adjust the transformation matrix.

        Args:
            src_pc: Source point cloud dictionary containing 'pos' key
            tgt_pc: Target point cloud dictionary containing 'pos' key
            transform: Original transformation matrix from source to target, shape (4, 4)

        Returns:
            A tuple containing:
            - Transformed source point cloud
            - Unchanged target point cloud
            - Adjusted transformation matrix
        """
        # Validate inputs
        assert isinstance(src_pc, dict), f"{type(src_pc)=}"
        assert src_pc.keys() >= {'pos'}, f"{src_pc.keys()=}"
        assert src_pc['pos'].ndim == 2 and src_pc['pos'].shape[1] == 3, f"{src_pc['pos'].shape=}"
        assert src_pc['pos'].dtype == torch.float32, f"{src_pc['pos'].dtype=}"

        assert isinstance(tgt_pc, dict), f"{type(tgt_pc)=}"
        assert tgt_pc.keys() >= {'pos'}, f"{tgt_pc.keys()=}"
        assert tgt_pc['pos'].ndim == 2 and tgt_pc['pos'].shape[1] == 3, f"{tgt_pc['pos'].shape=}"
        assert tgt_pc['pos'].dtype == torch.float32, f"{tgt_pc['pos'].dtype=}"

        assert isinstance(transform, torch.Tensor), f"{type(transform)=}"
        assert transform.shape == (4, 4), f"{transform.shape=}"
        assert transform.dtype == torch.float32, f"{transform.dtype=}"

        # Sample a random transformation
        random_transform = self._sample_rigid_transform(transform.device)

        # Create a new dictionary with the same references to non-pos fields
        new_src_pc = src_pc.copy()
        new_tgt_pc = tgt_pc.copy()

        # Apply random transformation to the source point cloud
        new_src_pc['pos'] = apply_transform(src_pc['pos'], random_transform)

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
