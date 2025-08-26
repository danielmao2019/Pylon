"""Generate change map between two point clouds.

This utility function computes change detection between two point clouds
based on nearest neighbor matching and segmentation labels.
"""

import torch
from typing import Dict
from sklearn.neighbors import NearestNeighbors


def generate_change_map(
    pc_from: Dict[str, torch.Tensor],
    pc_to: Dict[str, torch.Tensor],
    threshold: float,
    ignore_index: int = 255
) -> torch.Tensor:
    """Generate change map from source to target point cloud.
    
    For each point in pc_to, finds nearest neighbor in pc_from:
    - If distance < threshold and same segmentation: unchanged (0)
    - If distance < threshold and different segmentation: changed (1)
    - If distance >= threshold: ignore index
    
    Args:
        pc_from: Source point cloud (T1) with 'pos' and 'classification' keys
        pc_to: Target point cloud (T2) with 'pos' and 'classification' keys
        threshold: Distance threshold in meters
        ignore_index: Label for points with no valid match (default: 255)
        
    Returns:
        Change labels for each point in pc_to
    """
    # Extract positions and segmentation labels
    pos_from = pc_from['pos'].cpu().numpy()
    pos_to = pc_to['pos'].cpu().numpy()
    seg_from = pc_from['classification'].cpu().numpy()
    seg_to = pc_to['classification'].cpu().numpy()
    
    # Handle empty point clouds
    if pos_to.shape[0] == 0:
        return torch.zeros(0, dtype=torch.uint8, device=pc_to['pos'].device)
    
    if pos_from.shape[0] == 0:
        # No points in source -> all points are new (ignore index)
        return torch.full((pos_to.shape[0],), ignore_index, dtype=torch.uint8, device=pc_to['pos'].device)
    
    # Build KD-tree for nearest neighbor search
    nn_model = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
    nn_model.fit(pos_from)
    
    # Find nearest neighbors
    distances, indices = nn_model.kneighbors(pos_to)
    distances = distances.squeeze()
    indices = indices.squeeze()
    
    # Convert to torch tensors on correct device
    device = pc_to['pos'].device
    distances = torch.from_numpy(distances).to(device=device)
    indices = torch.from_numpy(indices).to(device=device)
    
    # Vectorized change label generation
    # Initialize all labels as ignore index on the correct device
    change_labels = torch.full((pos_to.shape[0],), ignore_index, dtype=torch.uint8, device=device)
    
    # Create mask for points within threshold
    within_threshold = distances < threshold
    
    # Convert numpy segmentation arrays to torch tensors on correct device
    seg_from_torch = torch.from_numpy(seg_from).to(device=device)
    seg_to_torch = torch.from_numpy(seg_to).to(device=device)
    
    # Get segmentation labels for matched points
    matched_seg_from = seg_from_torch[indices[within_threshold]]
    matched_seg_to = seg_to_torch[within_threshold]
    
    # Check if segmentation labels match
    seg_matches = matched_seg_from == matched_seg_to
    
    # Create indices for updating change_labels using torch operations
    within_threshold_indices = torch.where(within_threshold)[0]
    
    # Set unchanged (0) for matching segmentations
    unchanged_indices = within_threshold_indices[seg_matches]
    change_labels[unchanged_indices] = 0
    
    # Set changed (1) for non-matching segmentations
    changed_indices = within_threshold_indices[~seg_matches]
    change_labels[changed_indices] = 1
    
    return change_labels
