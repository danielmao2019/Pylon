"""
Pure PyTorch implementations of graph neural network operations.
These are equivalent to torch_geometric operations but implemented using native PyTorch.
"""

import torch
from typing import Optional, Tuple, Union, List
from torch import Tensor

def knn_graph(x: Tensor, k: int, batch: Optional[Tensor] = None, loop: bool = False, 
              flow: str = 'source_to_target', num_workers: int = 1) -> Tensor:
    """Computes graph edges to the nearest k points.
    
    Args:
        x: Node feature matrix of shape [N, F]
        k: The number of neighbors
        batch: Batch vector of shape [N] which assigns each node to a specific example
        loop: If True, the graph will contain self-loops
        flow: The flow direction when using in combination with message passing
        num_workers: Number of workers for parallel computation
        
    Returns:
        Edge index tensor of shape [2, N * k]
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
    # Compute pairwise distances
    dist = torch.cdist(x, x)
    
    # Remove self-loops if not requested
    if not loop:
        dist.fill_diagonal_(float('inf'))
        
    # Get k nearest neighbors
    _, index = dist.topk(k, largest=False)
    
    # Create edge index
    row = torch.arange(x.size(0), device=x.device).repeat_interleave(k)
    col = index.view(-1)
    
    if flow == 'source_to_target':
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = torch.stack([col, row], dim=0)
        
    return edge_index

def radius_graph(x: Tensor, r: float, batch: Optional[Tensor] = None, loop: bool = False,
                max_num_neighbors: int = 32, flow: str = 'source_to_target',
                num_workers: int = 1) -> Tensor:
    """Computes graph edges to all points within a given radius.
    
    Args:
        x: Node feature matrix of shape [N, F]
        r: The radius
        batch: Batch vector of shape [N] which assigns each node to a specific example
        loop: If True, the graph will contain self-loops
        max_num_neighbors: The maximum number of neighbors to return for each node
        flow: The flow direction when using in combination with message passing
        num_workers: Number of workers for parallel computation
        
    Returns:
        Edge index tensor of shape [2, E]
    """
    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
    # Compute pairwise distances
    dist = torch.cdist(x, x)
    
    # Remove self-loops if not requested
    if not loop:
        dist.fill_diagonal_(float('inf'))
        
    # Get points within radius
    mask = dist <= r
    
    # Limit number of neighbors if requested
    if max_num_neighbors is not None:
        # Sort distances for each node
        sorted_dist, sorted_idx = dist.sort(dim=1)
        # Keep only the closest max_num_neighbors
        mask = torch.zeros_like(mask)
        for i in range(x.size(0)):
            mask[i, sorted_idx[i, :max_num_neighbors]] = True
            
    # Create edge index
    row, col = mask.nonzero().t()
    
    if flow == 'source_to_target':
        edge_index = torch.stack([row, col], dim=0)
    else:
        edge_index = torch.stack([col, row], dim=0)
        
    return edge_index

def dense_to_sparse(x: Tensor) -> Tuple[Tensor, Tensor]:
    """Converts a dense adjacency matrix to a sparse adjacency matrix.
    
    Args:
        x: Dense adjacency matrix of shape [N, N]
        
    Returns:
        Tuple of (edge_index, edge_attr)
    """
    edge_index = x.nonzero().t()
    edge_attr = x[edge_index[0], edge_index[1]]
    return edge_index, edge_attr

def sparse_to_dense(edge_index: Tensor, edge_attr: Optional[Tensor] = None,
                   num_nodes: Optional[int] = None) -> Tensor:
    """Converts a sparse adjacency matrix to a dense adjacency matrix.
    
    Args:
        edge_index: Edge index tensor of shape [2, E]
        edge_attr: Edge feature tensor of shape [E]
        num_nodes: The number of nodes. If None, will be inferred from edge_index
        
    Returns:
        Dense adjacency matrix of shape [N, N]
    """
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
        
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)
        
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = edge_attr
    return adj 