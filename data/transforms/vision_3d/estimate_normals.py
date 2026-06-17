from typing import Any, Dict

import numpy as np
import open3d as o3d
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.transforms.base_transform import BaseTransform


class EstimateNormals(BaseTransform):

    def _call_single(self, pc: PointCloud) -> PointCloud:
        """
        Args:
            pc (Dict[str, Any]): The point cloud to estimate normals for.
        """
        assert isinstance(pc, PointCloud), f"{type(pc)=}"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.xyz.detach().cpu().numpy())
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location()
        pc.normals = torch.from_numpy(np.array(pcd.normals)).to(pc.xyz.device)
        return pc
