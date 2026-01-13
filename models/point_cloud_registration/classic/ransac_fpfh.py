from typing import Dict, List, Union

import torch
import numpy as np
import open3d as o3d

from data.structures.three_d.point_cloud.point_cloud import PointCloud


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

    def forward(
        self, inputs: Dict[str, Union[PointCloud, List[PointCloud]]]
    ) -> torch.Tensor:
        """Perform RANSAC-based registration using FPFH features.

        Args:
            inputs: Dictionary containing:
                - 'src_pc': PointCloud or list of PointClouds with source points
                - 'tgt_pc': PointCloud or list of PointClouds with target points

        Returns:
            Transformation matrix (B, 4, 4) that aligns source to target
        """
        assert isinstance(inputs, dict), "inputs must be a dictionary"
        assert 'src_pc' in inputs, "inputs must contain 'src_pc' key"
        assert 'tgt_pc' in inputs, "inputs must contain 'tgt_pc' key"
        src_pc = inputs['src_pc']
        tgt_pc = inputs['tgt_pc']
        src_list = src_pc if isinstance(src_pc, list) else [src_pc]
        tgt_list = tgt_pc if isinstance(tgt_pc, list) else [tgt_pc]

        assert len(src_list) > 0, "src_pc list must not be empty"
        assert len(tgt_list) > 0, "tgt_pc list must not be empty"
        assert len(src_list) == len(
            tgt_list
        ), f"Batch sizes must match: source={len(src_list)}, target={len(tgt_list)}"
        assert all(
            isinstance(pc, PointCloud) for pc in src_list
        ), f"src_pc entries must be PointClouds"
        assert all(
            isinstance(pc, PointCloud) for pc in tgt_list
        ), f"tgt_pc entries must be PointClouds"

        device = src_list[0].device
        assert all(
            pc.device == device for pc in src_list
        ), "All source point clouds must be on the same device"
        assert all(
            pc.device == device for pc in tgt_list
        ), "All target point clouds must be on the same device as sources"

        transformations = []
        for src_item, tgt_item in zip(src_list, tgt_list):
            source_o3d = o3d.geometry.PointCloud()
            source_o3d.points = o3d.utility.Vector3dVector(
                src_item.xyz.detach().cpu().numpy()
            )
            target_o3d = o3d.geometry.PointCloud()
            target_o3d.points = o3d.utility.Vector3dVector(
                tgt_item.xyz.detach().cpu().numpy()
            )

            # Downsample point clouds
            source_down = source_o3d.voxel_down_sample(self.voxel_size)
            target_down = target_o3d.voxel_down_sample(self.voxel_size)

            # Estimate normals
            source_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2, max_nn=30
                )
            )
            target_down.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 2, max_nn=30
                )
            )

            # Compute FPFH features
            fpfh_source = o3d.pipelines.registration.compute_fpfh_feature(
                source_down,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 5, max_nn=100
                ),
            )
            fpfh_target = o3d.pipelines.registration.compute_fpfh_feature(
                target_down,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel_size * 5, max_nn=100
                ),
            )

            # RANSAC registration
            reg_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down,
                target_down,
                fpfh_source,
                fpfh_target,
                mutual_filter=True,
                max_correspondence_distance=self.voxel_size * 1.5,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                ransac_n=4,
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                    100000, 0.999
                ),
            )

            transformations.append(reg_ransac.transformation)

        # Convert back to tensor
        return torch.tensor(
            np.stack(transformations), dtype=torch.float32, device=device
        )
