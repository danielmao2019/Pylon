import importlib


ext_module = importlib.import_module('models.point_cloud_registration.parenet.pareconv.ext')


def grid_subsample(points, lengths, voxel_size):
    """Grid subsampling in stack mode.

    This function is implemented on CPU but handles device transfers automatically.

    Args:
        points (Tensor): stacked points. (N, 3)
        lengths (Tensor): number of points in the stacked batch. (B,)
        voxel_size (float): voxel size.

    Returns:
        s_points (Tensor): stacked subsampled points (M, 3)
        s_lengths (Tensor): numbers of subsampled points in the batch. (B,)
    """
    # Store original device
    original_device = points.device
    
    # Move to CPU for C++ extension
    points_cpu = points.cpu()
    lengths_cpu = lengths.cpu()
    
    # Call C++ extension on CPU
    s_points_cpu, s_lengths_cpu = ext_module.grid_subsampling(points_cpu, lengths_cpu, voxel_size)
    
    # Move back to original device
    s_points = s_points_cpu.to(original_device)
    s_lengths = s_lengths_cpu.to(original_device)
    
    return s_points, s_lengths
