import torch


def radius_search(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit):
    r"""Computes neighbors for a batch of q_points and s_points, apply radius search (in stack mode).

    This function is implemented on CPU.

    Args:
        q_points (Tensor): the query points (N, 3)
        s_points (Tensor): the support points (M, 3)
        q_lengths (Tensor): the list of lengths of batch elements in q_points
        s_lengths (Tensor): the list of lengths of batch elements in s_points
        radius (float): maximum distance of neighbors
        neighbor_limit (int): maximum number of neighbors

    Returns:
        neighbors (Tensor): the k nearest neighbors of q_points in s_points (N, k).
            Filled with M if there are less than k neighbors.
    """
    import sys
    sys.path.append('./data/collators/geotransformer')
    import importlib
    ext_module = importlib.import_module('geotransformer.ext')

    device = q_points.device
    neighbor_indices = ext_module.radius_neighbors(q_points.cpu(), s_points.cpu(), q_lengths.cpu(), s_lengths.cpu(), radius)
    neighbor_indices = neighbor_indices.to(device)
    assert len(neighbor_indices) == len(q_points)

    # Create output tensor with correct shape and pad with invalid indices
    output = torch.full((q_points.shape[0], neighbor_limit), s_points.shape[0], dtype=neighbor_indices.dtype, device=device)

    # Copy valid neighbors and pad with s_points.shape[0] if needed
    if neighbor_indices.shape[1] > 0:
        output[:, :min(neighbor_indices.shape[1], neighbor_limit)] = neighbor_indices[:, :neighbor_limit]

    return output
