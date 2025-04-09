from typing import Tuple, Dict, Any
import torch
from data.transforms.base_transform import BaseTransform


class PCRTranslation(BaseTransform):
    """Translation transform that shifts point cloud positions to have mean at origin."""

    def __call__(
        self,
        src_pc: Dict[str, Any],
        tgt_pc: Dict[str, Any],
        transform: torch.Tensor,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor]:
        assert isinstance(src_pc, dict), f"{type(src_pc)=}"
        assert src_pc.keys() >= {'pos'}, f"{src_pc.keys()=}"
        assert src_pc['pos'].ndim == 2 and src_pc['pos'].shape[1] == 3, f"{src_pc['pos'].shape=}"
        assert isinstance(tgt_pc, dict), f"{type(tgt_pc)=}"
        assert tgt_pc.keys() >= {'pos'}, f"{tgt_pc.keys()=}"
        assert tgt_pc['pos'].ndim == 2 and tgt_pc['pos'].shape[1] == 3, f"{tgt_pc['pos'].shape=}"
        assert isinstance(transform, torch.Tensor), f"{type(transform)=}"
        assert transform.shape == (4, 4), f"{transform.shape=}"
        
        # Extract point positions from source and target point clouds
        src_pos = src_pc['pos']
        tgt_pos = tgt_pc['pos']
        
        # Calculate the mean of the union of both point clouds
        # First, concatenate the points
        union_points = torch.cat([src_pos, tgt_pos], dim=0)
        # Calculate the mean
        translation = union_points.mean(dim=0)
        
        # Create a copy of the point clouds to avoid modifying the originals
        new_src_pc = src_pc.copy()
        new_tgt_pc = tgt_pc.copy()
        
        # Apply translation to source and target point clouds
        new_src_pc['pos'] = src_pos - translation
        new_tgt_pc['pos'] = tgt_pos - translation
        
        # Adjust the transform to account for the translation
        # For a rigid transform T = [R|t], we need to adjust the translation part
        # The new translation is: t_new = t - (I - R) * translation
        # where I is the identity matrix and R is the rotation part of the transform
        
        # Extract rotation and translation from the transform
        # Assuming transform is a 4x4 matrix with rotation in the top-left 3x3 and translation in the last column
        R = transform[:3, :3]
        t = transform[:3, 3]
        
        # Calculate the new translation
        I = torch.eye(3, device=transform.device)
        new_t = t - (I - R) @ translation
        
        # Create the new transform
        new_transform = transform.clone()
        new_transform[:3, 3] = new_t
        
        return new_src_pc, new_tgt_pc, new_transform
