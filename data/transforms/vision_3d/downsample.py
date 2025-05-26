from typing import Dict
import torch
import open3d as o3d
from data.transforms.base_transform import BaseTransform


class DownSample(BaseTransform):

    def __init__(self, voxel_size: int):
        self.voxel_size = voxel_size

    def _call_single_(self, pc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(pc, dict)
        assert all(isinstance(k, str) for k in pc.keys())
        assert all(isinstance(v, torch.Tensor) for v in pc.values())

        assert 'pos' in pc, f"{pc.keys()=}"
        assert pc['pos'].ndim == 2, f"{pc['pos'].shape=}"
        assert pc['pos'].shape[1] == 3, f"{pc['pos'].shape=}"

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc['pos'].numpy())
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
