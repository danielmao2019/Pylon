import open3d as o3d
import numpy as np


def ransac_fpfh(source: np.ndarray, target: np.ndarray, voxel_size: float = 0.05) -> np.ndarray:
    source_o3d = o3d.geometry.PointCloud()
    source_o3d.points = o3d.utility.Vector3dVector(source)
    target_o3d = o3d.geometry.PointCloud()
    target_o3d.points = o3d.utility.Vector3dVector(target)
    
    source_down = source_o3d.voxel_down_sample(voxel_size)
    target_down = target_o3d.voxel_down_sample(voxel_size)
    
    source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    
    fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    
    reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, fpfh_source, fpfh_target,
        mutual_filter=True, max_correspondence_distance=voxel_size*1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=4,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return reg_ransac.transformation
