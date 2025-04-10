from typing import List, Dict
import torch
from utils.point_cloud_ops.sampling import GridSampling3D
from data.transforms.vision_3d.select import Select


def grid_sampling(
    pcs: List[Dict[str, torch.Tensor]],
    voxel_size: float,
) -> List[List[Dict[str, torch.Tensor]]]:
    """
    Grid sampling of point clouds.

    Args:
        pcs: List of point clouds
        voxel_size: Size of voxel cells for sampling

    Returns:
        A list of lists of voxelized point clouds, one list per input point cloud.
        Each voxelized point cloud contains the following fields:
        - pos: Point positions
        - indices: Original indices of the points
        - Other fields from the original point cloud, selected by indices
    """
    # Get the number of points in each point cloud
    num_points_per_pc = [pc['pos'].shape[0] for pc in pcs]

    # Create a list to keep track of the start index for each point cloud using cumsum
    start_indices = [0] + torch.cumsum(torch.tensor(num_points_per_pc[:-1]), dim=0).tolist()

    # Concatenate all point clouds
    points_union = torch.cat([pc['pos'] for pc in pcs], dim=0)

    # Center the points
    points_union = points_union - points_union.mean(dim=0, keepdim=True)

    # Create a grid sampler
    sampler = GridSampling3D(size=voxel_size)

    # Apply grid sampling to the union of all point clouds
    sampled_data = sampler({'pos': points_union})

    # Get the cluster indices for each point
    cluster_indices = sampled_data['point_indices']

    # Get unique cluster IDs
    unique_clusters = torch.unique(cluster_indices)

    # Initialize result lists
    result = [[] for _ in range(len(pcs))]

    # Process each cluster
    for cluster_id in unique_clusters:
        # Get points in this cluster
        cluster_mask = cluster_indices == cluster_id

        # Process each point cloud
        for pc_idx, pc in enumerate(pcs):
            # Create a mask for points from this point cloud
            pc_mask = torch.zeros_like(cluster_mask)
            pc_mask[start_indices[pc_idx]:start_indices[pc_idx] + num_points_per_pc[pc_idx]] = True

            # Get points in this cluster from this point cloud
            pc_cluster_mask = cluster_mask & pc_mask

            # If there are points in this cluster from this point cloud
            if pc_cluster_mask.any():
                # Get the indices of points in this cluster from this point cloud
                pc_cluster_indices = torch.where(pc_cluster_mask)[0]

                # Adjust indices to be relative to the original point cloud
                pc_cluster_indices = pc_cluster_indices - start_indices[pc_idx]

                # Use the Select transform to create a voxelized point cloud
                voxel_pc = Select(pc_cluster_indices)(pc)

                # Add the voxelized point cloud to the result
                result[pc_idx].append(voxel_pc)

    return result
