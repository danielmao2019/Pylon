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

        # Handle both 2D (unbatched) and 3D (batched) tensors
        if inputs['src_pc']['pos'].dim() == 2:
            inputs['src_pc']['pos'] = inputs['src_pc']['pos'].unsqueeze(0)
        if inputs['tgt_pc']['pos'].dim() == 2:
            inputs['tgt_pc']['pos'] = inputs['tgt_pc']['pos'].unsqueeze(0)

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

            # Run ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source_o3d, target_o3d, self.threshold, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.max_iterations)
            )

            transformations.append(reg_p2p.transformation)

        # Convert back to tensor
        return torch.tensor(np.stack(transformations), dtype=torch.float32, device=device)
