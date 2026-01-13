from typing import Any, Tuple
import torch
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class PCRTranslation(BaseTransform):
    """Translation transform that shifts point cloud positions to have mean at origin."""

    def __call__(
        self,
        src_pc: PointCloud,
        tgt_pc: PointCloud,
        transform: torch.Tensor,
    ) -> Tuple[PointCloud, PointCloud, torch.Tensor]:
        assert isinstance(src_pc, PointCloud), f"{type(src_pc)=}"
        assert isinstance(tgt_pc, PointCloud), f"{type(tgt_pc)=}"
        assert isinstance(transform, torch.Tensor), f"{type(transform)=}"
        assert transform.shape == (4, 4), f"{transform.shape=}"
        assert transform.device == src_pc.xyz.device, f"{transform.device=}, {src_pc.xyz.device=}"
        assert transform.dtype == src_pc.xyz.dtype, f"{transform.dtype=}, {src_pc.xyz.dtype=}"
        assert tgt_pc.xyz.device == src_pc.xyz.device, f"{tgt_pc.xyz.device=}, {src_pc.xyz.device=}"
        assert tgt_pc.xyz.dtype == src_pc.xyz.dtype, f"{tgt_pc.xyz.dtype=}, {src_pc.xyz.dtype=}"

        # Calculate the mean of the union of both point clouds
        # First, concatenate the points
        union_points = torch.cat([src_pc.xyz, tgt_pc.xyz], dim=0)
        # Calculate the mean
        translation = union_points.mean(dim=0)

        # Create new dictionaries with the same references to non-pos fields
        # Apply translation to source and target point clouds
        # This creates new tensors, so we don't need to clone
        src_pc.xyz = src_pc.xyz - translation
        tgt_pc.xyz = tgt_pc.xyz - translation

        # Adjust the transform to account for the translation
        # For a rigid transform T = [R|t], we need to adjust the translation part
        # The new translation is: t_new = t - (I - R) * translation
        # where I is the identity matrix and R is the rotation part of the transform

        # Extract rotation and translation from the transform
        # Assuming transform is a 4x4 matrix with rotation in the top-left 3x3 and translation in the last column
        R = transform[:3, :3]
        t = transform[:3, 3]

        # Calculate the new translation
        I = torch.eye(3, device=transform.device, dtype=transform.dtype)
        new_t = t - (I - R) @ translation

        # Create the new transform
        new_transform = transform.clone()
        new_transform[:3, 3] = new_t

        return src_pc, tgt_pc, new_transform
