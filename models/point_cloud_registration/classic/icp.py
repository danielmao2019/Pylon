import open3d as o3d
import numpy as np


def icp(source: np.ndarray, target: np.ndarray, threshold: float = 0.02, max_iterations: int = 50) -> np.ndarray:
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_o3d, target_o3d, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    return reg_p2p.transformation
