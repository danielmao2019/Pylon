from typing import Dict
import numpy as np
import torch
import open3d as o3d
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud
from utils.point_cloud_ops.select import Select


class DownSample(BaseTransform):

    def __init__(self, voxel_size: int):
        self.voxel_size = voxel_size

    def _call_single_(self, pc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        check_point_cloud(pc)

        # Convert to Open3D point cloud for downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc['pos'].detach().cpu().numpy())

        # Perform voxel downsampling and get indices directly
        _, _, kept_indices = pcd.voxel_down_sample_and_trace(
            voxel_size=self.voxel_size,
            min_bound=pcd.get_min_bound(),
            max_bound=pcd.get_max_bound()
        )

        # Create a Select operation with the kept indices
        select_op = Select(kept_indices)

        # Apply the Select operation to the original point cloud
        downsampled_pc = select_op(pc)

        return downsampled_pc
