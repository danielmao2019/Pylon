from typing import Dict, List, Union

import numpy as np
import open3d as o3d
import torch

from data.structures.three_d.point_cloud.point_cloud import PointCloud


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

    def forward(
        self, inputs: Dict[str, Union[PointCloud, List[PointCloud]]]
    ) -> torch.Tensor:
        """Perform ICP registration on source and target point clouds.

        Args:
            inputs: Dictionary containing:
                - 'src_pc': PointCloud or list of PointClouds with source points
                - 'tgt_pc': PointCloud or list of PointClouds with target points

        Returns:
            Transformation matrix (B, 4, 4) that aligns source to target

        Note:
            This implementation uses Open3D's ICP algorithm
        """
        # Input validation
        assert isinstance(inputs, dict), "inputs must be a dictionary"
        assert 'src_pc' in inputs, "inputs must contain 'src_pc' key"
        assert 'tgt_pc' in inputs, "inputs must contain 'tgt_pc' key"
        src_pc = inputs['src_pc']
        tgt_pc = inputs['tgt_pc']
        if isinstance(src_pc, list):
            src_list = src_pc
        else:
            src_list = [src_pc]
        if isinstance(tgt_pc, list):
            tgt_list = tgt_pc
        else:
            tgt_list = [tgt_pc]

        assert len(src_list) > 0, "src_pc list must not be empty"
        assert len(tgt_list) > 0, "tgt_pc list must not be empty"
        assert len(src_list) == len(
            tgt_list
        ), f"Batch sizes must match: source={len(src_list)}, target={len(tgt_list)}"
        assert all(
            isinstance(pc, PointCloud) for pc in src_list
        ), f"src_pc must be PointCloud or list of PointClouds, got {type(src_pc)}"
        assert all(
            isinstance(pc, PointCloud) for pc in tgt_list
        ), f"tgt_pc must be PointCloud or list of PointClouds, got {type(tgt_pc)}"
        device = src_list[0].device
        assert all(
            pc.device == device for pc in src_list
        ), "All source point clouds must be on the same device"
        assert all(
            pc.device == device for pc in tgt_list
        ), "All target point clouds must be on the same device"

        # Process each batch
        transformations = []
        for src_item, tgt_item in zip(src_list, tgt_list, strict=True):
            # Create Open3D point clouds
            source_o3d = o3d.geometry.PointCloud()
            source_o3d.points = o3d.utility.Vector3dVector(
                src_item.xyz.detach().cpu().numpy()
            )
            target_o3d = o3d.geometry.PointCloud()
            target_o3d.points = o3d.utility.Vector3dVector(
                tgt_item.xyz.detach().cpu().numpy()
            )

            # Run ICP
            reg_p2p = o3d.pipelines.registration.registration_icp(
                source=source_o3d,
                target=target_o3d,
                max_correspondence_distance=self.threshold,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=self.max_iterations
                ),
            )

            transformations.append(reg_p2p.transformation)

        # Convert back to tensor
        return torch.tensor(
            np.stack(transformations), dtype=torch.float32, device=device
        )
