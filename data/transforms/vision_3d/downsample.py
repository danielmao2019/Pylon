from typing import Union
import torch
import open3d as o3d
from data.transforms.base_transform import BaseTransform
from data.structures.three_d.point_cloud.select import Select
from data.structures.three_d.point_cloud.point_cloud import PointCloud


class DownSample(BaseTransform):

    def __init__(self, voxel_size: Union[float, int]):
        assert isinstance(voxel_size, (float, int))
        assert voxel_size > 0, f"{voxel_size=}"
        self.voxel_size = float(voxel_size)

    def _call_single(self, pc: PointCloud) -> PointCloud:
        """
        Args:
            pc (PointCloud): The point cloud to downsample.
        """
        assert isinstance(pc, PointCloud), f"{type(pc)=}"

        # Convert to Open3D point cloud for downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.xyz.detach().cpu().numpy())

        # Perform voxel downsampling and get indices directly
        _, _, kept_indices = pcd.voxel_down_sample_and_trace(
            voxel_size=self.voxel_size,
            min_bound=pcd.get_min_bound(),
            max_bound=pcd.get_max_bound(),
        )

        # Convert IntVector list to a list of integers (taking first point from each voxel)
        kept_indices = [indices[0] for indices in kept_indices]

        # Create a Select operation with the kept indices
        select_op = Select(kept_indices)

        # Apply the Select operation to the original point cloud
        downsampled_pc = select_op(pc)

        return downsampled_pc
