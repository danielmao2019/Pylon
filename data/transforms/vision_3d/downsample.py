from typing import Dict
import torch
import open3d as o3d
from data.transforms.base_transform import BaseTransform
from utils.input_checks.point_cloud import check_point_cloud


class DownSample(BaseTransform):

    def __init__(self, voxel_size: int):
        self.voxel_size = voxel_size

    def _call_single_(self, pc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        check_point_cloud(pc)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc['pos'].detach().cpu().numpy())
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
