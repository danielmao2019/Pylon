"""
Utility functions for SiameseKPConv model
"""
import torch


def add_ones(pos, x, add_one=True):
    """
    Add a ones feature to the input features x
    
    Args:
        pos: input positions [N, 3]
        x: input features [N, C]
        add_one: whether to add a constant feature
        
    Returns:
        features with an additional ones feature [N, C+1] if add_one is True
    """
    if add_one:
        ones = torch.ones(pos.shape[0], dtype=torch.float).unsqueeze(-1).to(pos.device)
        if x is not None:
            x = torch.cat([ones.to(x.dtype), x], dim=-1)
        else:
            x = ones
    return x


def gather(x, idx):
    """
    Gathers values from x according to indices idx.
    
    Args:
        x: input tensor [N, C] or [B, N, C]
        idx: indices [N', M] or [B, N', M]
        
    Returns:
        gathered values [N', M, C] or [B, N', M, C]
    """
    batch_size, num_points, num_dims = x.shape if len(x.shape) == 3 else (1, x.shape[0], x.shape[1])
    idx_flattened = idx + torch.arange(batch_size, device=idx.device).view(-1, 1, 1) * num_points
    return x.reshape(batch_size * num_points, num_dims)[idx_flattened.reshape(-1)].reshape(idx.shape + (num_dims,)) 