import torch
from models.point_cloud_registration.parenet.pareconv.extensions.pointops.functions import pointops


def radius_search(q_points, s_points, q_lengths, s_lengths, num_neighbors):
    r"""Computes k nearest neighbors for a batch of q_points and s_points.

    This function is implemented on GPU and requires CUDA tensors.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        num_neighbors (int): number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
    """
    # Store original device
    original_device = q_points.device
    
    # Move to CUDA if not already (C++ extension requires CUDA)
    if not q_points.is_cuda:
        q_points = q_points.cuda()
        s_points = s_points.cuda()
        q_lengths = q_lengths.cuda()
        s_lengths = s_lengths.cuda()
    
    q_pcd1 = q_points[:q_lengths[0]].unsqueeze(0)
    q_pcd2 = q_points[q_lengths[0]:].unsqueeze(0)

    s_pcd1 = s_points[:s_lengths[0]].unsqueeze(0)
    s_pcd2 = s_points[s_lengths[0]:].unsqueeze(0)

    ind_local1 = pointops.knnquery_heap(num_neighbors, s_pcd1, q_pcd1)
    ind_local2 = pointops.knnquery_heap(num_neighbors, s_pcd2, q_pcd2)

    ind_local2 = ind_local2 + s_lengths[0]
    index = torch.cat([ind_local1, ind_local2], 1)
    result = index.squeeze(0)
    
    # Move back to original device
    if not original_device.type == 'cuda':
        result = result.to(original_device)
    
    return result
