from typing import Dict
import torch
import numpy as np
import open3d as o3d


class RANSAC_FPFH(torch.nn.Module):
    """RANSAC-based registration using Fast Point Feature Histograms (FPFH).

    This method first downsamples point clouds, computes FPFH descriptors,
    then uses RANSAC to robustly estimate the transformation from feature correspondences.

    Args:
        voxel_size: Voxel size for downsampling and feature computation
    """

    def __init__(self, voxel_size: float = 0.05) -> None:
        """Initialize RANSAC-FPFH model with voxel size parameter.

        Args:
            voxel_size: Voxel size for downsampling. Also used to determine
                       radius for normal estimation (2x voxel_size) and
                       FPFH computation (5x voxel_size)
        """
        super(RANSAC_FPFH, self).__init__()
        self.voxel_size = voxel_size

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Perform RANSAC-based registration using FPFH features.

        Args:
            inputs: Dictionary containing:
                - 'src_pc': Dict with 'pos' key containing source points (B, N, 3)
                - 'tgt_pc': Dict with 'pos' key containing target points (B, M, 3)

        Returns:
            Transformation matrix (B, 4, 4) that aligns source to target
        """
        # Input validation
        assert isinstance(inputs, dict), "inputs must be a dictionary"
        assert 'src_pc' in inputs, "inputs must contain 'src_pc' key"
        assert 'tgt_pc' in inputs, "inputs must contain 'tgt_pc' key"
        assert isinstance(inputs['src_pc'], dict), "inputs['src_pc'] must be a dictionary"
        assert isinstance(inputs['tgt_pc'], dict), "inputs['tgt_pc'] must be a dictionary"
        assert 'pos' in inputs['src_pc'], "inputs['src_pc'] must contain 'pos' key"
        assert 'pos' in inputs['tgt_pc'], "inputs['tgt_pc'] must contain 'pos' key"
        
        # Validate tensor properties
        assert isinstance(inputs['src_pc']['pos'], torch.Tensor), "Source positions must be a tensor"
        assert isinstance(inputs['tgt_pc']['pos'], torch.Tensor), "Target positions must be a tensor"
        assert inputs['src_pc']['pos'].dim() == 3, f"Source positions must be 3D tensor, got {inputs['src_pc']['pos'].dim()}D"
        assert inputs['tgt_pc']['pos'].dim() == 3, f"Target positions must be 3D tensor, got {inputs['tgt_pc']['pos'].dim()}D"
        assert inputs['src_pc']['pos'].shape[-1] == 3, f"Source positions must have 3 coordinates, got {inputs['src_pc']['pos'].shape[-1]}"
        assert inputs['tgt_pc']['pos'].shape[-1] == 3, f"Target positions must have 3 coordinates, got {inputs['tgt_pc']['pos'].shape[-1]}"
        assert inputs['src_pc']['pos'].shape[0] == inputs['tgt_pc']['pos'].shape[0], \
            f"Batch sizes must match: source={inputs['src_pc']['pos'].shape[0]}, target={inputs['tgt_pc']['pos'].shape[0]}"
        assert inputs['src_pc']['pos'].shape[1] > 0, "Source point cloud cannot be empty"
        assert inputs['tgt_pc']['pos'].shape[1] > 0, "Target point cloud cannot be empty"
        
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
