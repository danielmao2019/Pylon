from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import torch

from data.structures.three_d.point_cloud.ops.sampling.grid_sampling_3d_v2 import (
    GridSampling3D,
)
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.select import Select


def process_cluster(
    args: Tuple[int, torch.Tensor, List[PointCloud], List[int], List[torch.Tensor]],
) -> Tuple[int, List[Optional[PointCloud]]]:
    """
    Process a single cluster for all point clouds.

    Args:
        args: Tuple containing (cluster_id, cluster_mask, pcs, start_indices, pc_masks)

    Returns:
        Tuple of (cluster_id, results) where results is a list of voxelized point clouds
    """
    # Input validations
    assert isinstance(args, tuple), f"{type(args)=}"
    assert len(args) == 5, f"{len(args)=}"

    cluster_id, cluster_mask, pcs, start_indices, pc_masks = args

    # Process each point cloud for this cluster
    results: List[Optional[PointCloud]] = []
    for pc_idx in range(len(pcs)):
        # Get points in this cluster from this point cloud using pre-computed mask
        pc_cluster_mask = cluster_mask & pc_masks[pc_idx]

        # If there are points in this cluster from this point cloud
        if pc_cluster_mask.any():
            # Get the indices of points in this cluster from this point cloud
            pc_cluster_indices = torch.where(pc_cluster_mask)[0]

            # Adjust indices to be relative to the original point cloud
            pc_cluster_indices = pc_cluster_indices - start_indices[pc_idx]

            # Use the Select transform to create a voxelized point cloud
            voxel_pc = Select(pc_cluster_indices)(pcs[pc_idx])

            # Add the voxelized point cloud to the result
            results.append(voxel_pc)
        else:
            results.append(None)

    return cluster_id, results


def grid_sampling(
    pcs: List[PointCloud],
    voxel_size: float,
    num_workers: Optional[int] = None,
) -> List[List[Optional[PointCloud]]]:
    """
    Grid sampling of point clouds.

    Args:
        pcs: List of point clouds
        voxel_size: Size of voxel cells for sampling
        num_workers: Number of worker processes to use for parallel processing.
                    If None, uses all available CPU cores.

    Returns:
        A list of lists of voxelized point clouds, one list per input point cloud.
        Each voxelized point cloud contains the following fields:
        - xyz: Point positions
        - indices: Original indices of the points
        - Other fields from the original point cloud, selected by indices
    """
    # Get the number of points in each point cloud
    num_points_per_pc = [pc.xyz.shape[0] for pc in pcs]

    # Create a list to keep track of the start index for each point cloud using cumsum
    start_indices = [0] + torch.cumsum(
        torch.tensor(num_points_per_pc[:-1]), dim=0
    ).tolist()

    # Concatenate all point clouds
    points_union = torch.cat([pc.xyz for pc in pcs], dim=0)

    # Center the points
    points_union = points_union - points_union.mean(dim=0, keepdim=True)

    sampler = GridSampling3D(size=voxel_size)

    sampled_data = sampler(PointCloud(xyz=points_union))

    cluster_indices = sampled_data.point_indices

    # Get unique cluster IDs
    unique_clusters = torch.unique(cluster_indices)

    # Initialize result lists
    result = [[] for _ in range(len(pcs))]

    # Create a tensor to track which point cloud each point belongs to
    pc_indices = torch.zeros(
        len(points_union), dtype=torch.long, device=points_union.device
    )
    for i, (start, num_points) in enumerate(
        zip(start_indices, num_points_per_pc, strict=True)
    ):
        pc_indices[start : start + num_points] = i

    # Pre-compute masks for each point cloud to avoid redundant calculations
    pc_masks = []
    for pc_idx in range(len(pcs)):
        pc_masks.append(pc_indices == pc_idx)

    # Prepare arguments for parallel processing
    process_args = [
        (
            cluster_id.item(),
            cluster_indices == cluster_id,
            pcs,
            start_indices,
            pc_masks,
        )
        for cluster_id in unique_clusters
    ]

    # Process clusters in parallel using ProcessPoolExecutor
    cluster_results = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(process_cluster, args): args for args in process_args
        }

        # Process results as they complete
        for future in as_completed(future_to_args):
            # This will raise any exceptions that occurred in the worker process
            cluster_id, cluster_result = future.result()
            cluster_results[cluster_id] = cluster_result

    # Reconstruct the result in the original order
    for cluster_id in unique_clusters:
        cluster_id = cluster_id.item()
        for pc_idx in range(len(pcs)):
            result[pc_idx].append(cluster_results[cluster_id][pc_idx])

    assert all(len(r) == len(result[0]) for r in result)
    return result
