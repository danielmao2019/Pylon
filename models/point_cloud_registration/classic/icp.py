from typing import Dict
import torch
import numpy as np
import open3d as o3d


class ICP(torch.nn.Module):
    """Iterative Closest Point (ICP) algorithm for point cloud registration.

    ICP iteratively refines the transformation between source and target point clouds
    by minimizing the distance between corresponding points.

    Args:
        threshold: Maximum correspondence distance threshold for convergence
        max_iterations: Maximum number of iterations for ICP optimization
    """

    def __init__(self, threshold: float = 0.02, max_iterations: int = 50) -> None:
        """Initialize ICP model with convergence parameters.

        Args:
            threshold: Maximum correspondence distance threshold for convergence
            max_iterations: Maximum number of iterations for ICP optimization
        """
        super(ICP, self).__init__()
        self.threshold = threshold
        self.max_iterations = max_iterations

    def forward(self, inputs: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Perform ICP registration on source and target point clouds.

        Args:
            inputs: Dictionary containing:
                - 'src_pc': Dict with 'pos' key containing source points (B, N, 3)
                - 'tgt_pc': Dict with 'pos' key containing target points (B, M, 3)

        Returns:
            Transformation matrix (B, 4, 4) that aligns source to target

        Note:
            This implementation uses Open3D's ICP algorithm
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

            # Run ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_o3d, target_o3d, self.threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
            )

            transformations.append(reg_p2p.transformation)

        # Convert back to tensor
        return torch.tensor(np.stack(transformations), dtype=torch.float32, device=device)
