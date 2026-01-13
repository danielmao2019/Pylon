"""Generate change map between two point clouds.

This utility function computes change detection between two point clouds
based on nearest neighbor matching and segmentation labels.
"""

from typing import Dict, Optional
import torch
from data.structures.three_d.point_cloud.point_cloud import PointCloud
from data.structures.three_d.point_cloud.ops.knn.knn import knn


def generate_change_map(
    src_pc: PointCloud,
    tgt_pc: PointCloud,
    threshold: float,
    seg_ignore_value: int = 255,
    change_ignore_value: int = 255,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Generate change map from source to target point cloud.

    Computes change detection labels for each point in the target point cloud
    by finding nearest neighbors in the source point cloud and comparing
    their segmentation labels.

    Change Detection Logic:
    -----------------------
    For each point in target point cloud:
    1. If target point has ignore segmentation → label as ignore (change_ignore_value)
    2. Find nearest neighbor in source point cloud (excluding source ignore points)
    3. If no valid source points exist → label as changed (1)
    4. If nearest neighbor distance >= threshold → label as changed (1)
    5. If distance < threshold:
       - Same segmentation class → label as unchanged (0)
       - Different segmentation class → label as changed (1)

    Args:
        src_pc: Source point cloud (T1) containing:
            - 'xyz': [N_src, 3] coordinates
            - 'classification': [N_src] segmentation labels
        tgt_pc: Target point cloud (T2) containing:
            - 'xyz': [N_tgt, 3] coordinates
            - 'classification': [N_tgt] segmentation labels
        threshold: Distance threshold in meters for matching points
        seg_ignore_value: Segmentation label to ignore (default: 255)
        change_ignore_value: Output label for ignored points (default: 255)
        chunk_size: Optional batch size for KNN computation to save memory

    Returns:
        torch.Tensor: Change labels [N_tgt] with values:
            - 0: unchanged (matched point with same segmentation)
            - 1: changed (no match or different segmentation)
            - change_ignore_value: ignored points
    """
    assert isinstance(src_pc, PointCloud), f"{type(src_pc)=}"
    assert isinstance(tgt_pc, PointCloud), f"{type(tgt_pc)=}"

    assert hasattr(src_pc, 'classification'), "Source point cloud missing classification field"
    assert hasattr(tgt_pc, 'classification'), "Target point cloud missing classification field"
    src_pos = src_pc.xyz  # [N_src, 3] - 3D positions
    tgt_pos = tgt_pc.xyz  # [N_tgt, 3] - 3D positions
    src_seg = getattr(src_pc, 'classification')  # [N_src] - segmentation labels
    tgt_seg = getattr(tgt_pc, 'classification')  # [N_tgt] - segmentation labels

    device = tgt_pos.device

    # Step 2: Handle edge case - empty target point cloud
    if tgt_pos.shape[0] == 0:
        return torch.zeros(0, dtype=torch.uint8, device=device)

    # Step 3: Initialize all target points as "changed" (conservative default)
    change_labels = torch.ones(tgt_pos.shape[0], dtype=torch.int64, device=device)

    # Step 4: Mark target points with ignore segmentation
    tgt_ignore_mask = tgt_seg == seg_ignore_value
    change_labels[tgt_ignore_mask] = change_ignore_value

    # Step 5: Handle edge case - empty or all-ignore source point cloud
    if src_pos.shape[0] == 0:
        # No source points available for matching
        return change_labels  # All non-ignore targets remain "changed"

    # Step 6: Filter out ignore points from source (only match to valid points)
    src_valid_mask = src_seg != seg_ignore_value
    if not src_valid_mask.any():
        # No valid source points for matching
        return change_labels  # All non-ignore targets remain "changed"

    src_pos_valid = src_pos[src_valid_mask]  # Valid source positions
    src_seg_valid = src_seg[src_valid_mask]  # Valid source segmentations

    # Step 7: Find nearest neighbor in source for each target point
    distances, indices = knn(
        query_points=tgt_pos,
        reference_points=src_pos_valid,
        k=1,  # Only need closest match
        method="pytorch3d",
        return_distances=True,
        chunk_size=chunk_size,  # Optional memory optimization
    )

    # Reshape results (remove k=1 dimension)
    distances = distances.squeeze(1)  # [N_tgt] - distance to nearest source point
    indices = indices.squeeze(1)  # [N_tgt] - index of nearest source point

    # Step 8: Identify target points with valid matches (within threshold)
    # Note: Exclude already-ignored target points from this check
    within_threshold = (distances < threshold) & (~tgt_ignore_mask)

    # Step 9: Compare segmentations for matched points
    if within_threshold.any():
        # Extract segmentation labels for matched point pairs
        matched_src_seg = src_seg_valid[indices[within_threshold]]
        matched_tgt_seg = tgt_seg[within_threshold]

        # Check if segmentation classes match
        seg_matches = matched_src_seg == matched_tgt_seg

        # Get global indices of target points within threshold
        within_threshold_indices = torch.where(within_threshold)[0]

        # Step 10: Update labels for unchanged points (same segmentation)
        unchanged_indices = within_threshold_indices[seg_matches]
        change_labels[unchanged_indices] = 0  # Mark as unchanged

        # Note: Points with different segmentations remain "changed" (1)

    return change_labels
