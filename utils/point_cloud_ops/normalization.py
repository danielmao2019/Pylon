import torch


def normalize_point_cloud(pc: torch.Tensor) -> torch.Tensor:
    """Normalize point cloud to unit sphere.
    
    Centers the point cloud at the origin and scales it so that the maximum
    distance from the origin is 1.0. This is commonly used for ModelNet40
    and other object-centric datasets.
    
    Args:
        pc: Point cloud tensor of shape (N, 3)
        
    Returns:
        Normalized point cloud tensor centered at origin with max distance = 1.0
    """
    if pc.numel() == 0:
        return pc
    
    # Center at origin
    pc_centered = pc - pc.mean(dim=0, keepdim=True)
    
    # Scale to unit sphere (max distance from origin = 1.0)
    max_dist = torch.norm(pc_centered, dim=1).max()
    if max_dist > 0:
        pc_normalized = pc_centered / max_dist
    else:
        pc_normalized = pc_centered
        
    return pc_normalized
