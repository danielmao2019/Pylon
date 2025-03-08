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


def knn(x, y, k, batch_x=None, batch_y=None):
    """
    Find k-nearest neighbors of points in x from points in y
    
    Args:
        x: tensor of shape [n, d] for n points with d dimensions
        y: tensor of shape [m, d] for m points with d dimensions
        k: number of nearest neighbors to find
        batch_x: optional batch indices for x
        batch_y: optional batch indices for y
        
    Returns:
        tuple of (row_idx, col_idx)
        row_idx: indices of query points in x
        col_idx: indices of nearest neighbors in y
    """
    if batch_x is None:
        batch_x = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    if batch_y is None:
        batch_y = torch.zeros(y.size(0), dtype=torch.long, device=y.device)
        
    row_idx = []
    col_idx = []
    
    # Process each batch separately
    for b in torch.unique(batch_x):
        x_mask = batch_x == b
        y_mask = batch_y == b
        
        x_b = x[x_mask]
        y_b = y[y_mask]
        
        # Compute squared distances
        # Using ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y for efficiency
        x_norm = (x_b ** 2).sum(1).view(-1, 1)
        y_norm = (y_b ** 2).sum(1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.matmul(x_b, y_b.transpose(0, 1))
        
        # Get k nearest neighbors
        _, nn_idx = torch.topk(dist, k=min(k, y_b.size(0)), dim=1, largest=False)
        
        # Convert to global indices
        rows = torch.nonzero(x_mask, as_tuple=True)[0]
        cols = torch.nonzero(y_mask, as_tuple=True)[0][nn_idx.reshape(-1)].reshape(rows.size(0), -1)
        
        row_idx.append(rows.repeat_interleave(nn_idx.size(1)))
        col_idx.append(cols.reshape(-1))
    
    # Concatenate results from all batches
    if len(row_idx) > 0:
        row_idx = torch.cat(row_idx)
        col_idx = torch.cat(col_idx)
    else:
        row_idx = torch.tensor([], dtype=torch.long, device=x.device)
        col_idx = torch.tensor([], dtype=torch.long, device=x.device)
    
    return row_idx, col_idx
