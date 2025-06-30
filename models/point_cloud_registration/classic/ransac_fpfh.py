from typing import Dict
import torch
import numpy as np
import open3d as o3d


class RANSAC_FPFH(torch.nn.Module):
    def __init__(self, voxel_size: float = 0.05):
        super(RANSAC_FPFH, self).__init__()
        self.voxel_size = voxel_size

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        RANSAC-based registration using FPFH features.

        Args:
            inputs: Dictionary containing source and target point clouds
            inputs['src_pc']['pos']: Source point cloud (B, N, 3)
            inputs['tgt_pc']['pos']: Target point cloud (B, M, 3)

        Returns:
            Transformation matrix (B, 4, 4)
        """
        batch_size = inputs['src_pc']['pos'].shape[0]
        device = inputs['src_pc']['pos'].device

        # Convert to numpy for Open3D
        source_np = inputs['src_pc']['pos'].detach().cpu().numpy()
        target_np = inputs['tgt_pc']['pos'].detach().cpu().numpy()

        # Process each batch
        transformations = []
        for i in range(batch_size):
            # Create Open3D point clouds
            source_o3d = o3d.geometry.PointCloud()
            source_o3d.points = o3d.utility.Vector3dVector(source_np[i])
            target_o3d = o3d.geometry.PointCloud()
            target_o3d.points = o3d.utility.Vector3dVector(target_np[i])

            # Downsample point clouds
            source_down = source_o3d.voxel_down_sample(self.voxel_size)
            target_down = target_o3d.voxel_down_sample(self.voxel_size)

            # Estimate normals
            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*2, max_nn=30))

            # Compute FPFH features
            fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
                source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))
            fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size*5, max_nn=100))

            # RANSAC registration
            reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, fpfh_source, fpfh_target,
                mutual_filter=True, max_correspondence_distance=self.voxel_size*1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=4,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

            transformations.append(reg_ransac.transformation)

        # Convert back to tensor
        return torch.tensor(np.stack(transformations), dtype=torch.float32, device=device)
