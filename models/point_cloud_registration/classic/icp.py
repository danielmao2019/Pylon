import torch
import torch.nn as nn
import numpy as np
import open3d as o3d


class ICPModule(nn.Module):
    def __init__(self, threshold: float = 0.02, max_iterations: int = 50):
        super().__init__()
        self.threshold = threshold
        self.max_iterations = max_iterations
        
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Iterative Closest Point (ICP) registration.
        
        Args:
            source: Source point cloud (B, N, 3)
            target: Target point cloud (B, M, 3)
            
        Returns:
            Transformation matrix (B, 4, 4)
        """
        batch_size = source.shape[0]
        device = source.device
        
        # Convert to numpy for Open3D
        source_np = source.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # Process each batch
        transformations = []
        for i in range(batch_size):
            # Create Open3D point clouds
            source_o3d = o3d.geometry.PointCloud()
            source_o3d.points = o3d.utility.Vector3dVector(source_np[i])
            target_o3d = o3d.geometry.PointCloud()
            target_o3d.points = o3d.utility.Vector3dVector(target_np[i])
            
            # Run ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_o3d, target_o3d, self.threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
            )
            
            transformations.append(reg_p2p.transformation)
        
        # Convert back to tensor
        return torch.tensor(np.stack(transformations), device=device)
