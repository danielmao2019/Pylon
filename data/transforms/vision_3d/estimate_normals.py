from typing import Dict, Any
import numpy as np
import torch
import open3d as o3d
from data.transforms.base_transform import BaseTransform
from utils.input_checks import check_point_cloud


class EstimateNormals(BaseTransform):

    def _call_single(self, pc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            pc (Dict[str, Any]): The point cloud to estimate normals for.
        """
        check_point_cloud(pc)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc['pos'].detach().cpu().numpy())
        pcd.estimate_normals()
        pcd.orient_normals_towards_camera_location()
        pc['normals'] = torch.from_numpy(np.array(pcd.normals)).to(pc['pos'].device)
        return pc
