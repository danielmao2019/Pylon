from typing import Dict, Any, Tuple
import torch
import numpy as np
from data.transforms.base_transform import BaseTransform
from utils.point_cloud_ops import apply_transform


class RandomRigidTransform(BaseTransform):
    """Random rigid transformation (rotation and translation) for point clouds."""

    def __init__(self, rot_mag: float = 45.0, trans_mag: float = 0.5):
        """
        Initialize the random rigid transform.

        Args:
            rot_mag: Maximum rotation magnitude in degrees (default: 45.0)
            trans_mag: Maximum translation magnitude (default: 0.5)
        """
        super(RandomRigidTransform, self).__init__()
        assert isinstance(rot_mag, (int, float)), f"{type(rot_mag)=}"
        assert isinstance(trans_mag, (int, float)), f"{type(trans_mag)=}"
        assert rot_mag >= 0, f"{rot_mag=}"
        assert trans_mag >= 0, f"{trans_mag=}"
        self.rot_mag = rot_mag
        self.trans_mag = trans_mag

    def _sample_random_transform(self, device: torch.device) -> torch.Tensor:
        """
        Sample a random rigid transformation.

        Args:
            device: Device to create tensors on

        Returns:
            A 4x4 transformation matrix
        """
        # Generate random transformation
        rot_mag_rad = np.radians(self.rot_mag)

        # Generate a random axis of rotation
        axis = torch.randn(3, device=device)
        axis = axis / torch.norm(axis)  # Normalize to unit vector

        # Generate random angle within the specified range
        angle = torch.empty(1, device=device).uniform_(-rot_mag_rad, rot_mag_rad)

        # Create rotation matrix using axis-angle representation (Rodrigues' formula)
        K = torch.tensor([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]], dtype=torch.float32, device=device)
        R = torch.eye(3, device=device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        # Generate random translation
        # Generate random direction (unit vector)
        trans_dir = torch.randn(3, device=device)
        trans_dir = trans_dir / torch.norm(trans_dir)

        # Generate random magnitude within limit
        trans_mag = torch.empty(1, device=device).uniform_(0, self.trans_mag)

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
        random_transform = self._sample_random_transform(transform.device)

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
        assert torch.equal(random_transform_inv[-1, :], torch.tensor([0, 0, 0, 1], device=random_transform_inv.device))
        assert torch.allclose(random_transform_inv[:3, :3], random_transform[:3, :3].T), f"{random_transform_inv[:3, :3]=}\n{random_transform[:3, :3].T=}"
        assert torch.allclose(random_transform_inv[:3, 3], -random_transform[:3, :3].T @ random_transform[:3, 3])

        new_transform = transform @ random_transform_inv

        return new_src_pc, new_tgt_pc, new_transform
