# Multi-View Fusion with Voting

## Overview
Multi-view fusion is the most straightforward approach for projecting 2D segmentation results to 3D point clouds. It leverages multiple camera viewpoints to obtain segmentation predictions for each point, then aggregates these predictions through voting or confidence-weighted averaging.

## Technical Approach

### 1. Forward Projection Pipeline
```python
def multiview_segmentation_fusion(
    point_cloud: torch.Tensor,           # [N, 3] 3D point coordinates
    point_features: Dict[str, torch.Tensor],  # Additional features (rgb, etc.)
    camera_intrinsics: List[torch.Tensor],    # List of [3, 3] matrices
    camera_extrinsics: List[torch.Tensor],    # List of [4, 4] matrices
    images: List[torch.Tensor],               # List of [3, H, W] images
    segmentation_model_2d,                    # 2D segmentation model
    aggregation_method: str = "majority_vote" # or "confidence_weighted", "bayesian"
) -> Dict[str, torch.Tensor]:
    """
    Project 2D segmentation to 3D points using multi-view fusion.
    
    Returns:
        Dictionary containing:
        - 'labels': [N] predicted class indices
        - 'confidence': [N] prediction confidence scores
        - 'vote_matrix': [N, num_classes] raw vote counts
    """
```

### 2. Core Algorithm Steps

#### Step 1: Visibility Computation
For each camera view, determine which points are visible:
```python
def compute_visibility(
    points: torch.Tensor,          # [N, 3]
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    image_shape: Tuple[int, int],
    occlusion_threshold: float = 0.01
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Transform to camera coordinates
    world_to_camera = torch.inverse(camera_extrinsics)
    points_cam = transform_points(points, world_to_camera)
    
    # Check if points are in front of camera
    depth_mask = points_cam[:, 2] > 0
    
    # Project to image plane
    points_2d = project_to_image(points_cam, camera_intrinsics)
    
    # Check if within image bounds
    bounds_mask = (
        (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_shape[1]) &
        (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_shape[0])
    )
    
    # Z-buffer for occlusion handling
    depth_buffer = render_depth_buffer(points_cam, points_2d, image_shape)
    occlusion_mask = check_occlusion(points_cam, points_2d, depth_buffer, occlusion_threshold)
    
    visibility_mask = depth_mask & bounds_mask & ~occlusion_mask
    
    return points_2d, points_cam[:, 2], visibility_mask


def render_depth_buffer(
    points_cam: torch.Tensor,      # [N, 3] points in camera coordinates
    points_2d: torch.Tensor,        # [N, 2] projected 2D coordinates
    image_shape: Tuple[int, int]   # (H, W) image dimensions
) -> torch.Tensor:
    """
    Create a depth buffer for occlusion testing.
    
    The depth buffer stores the minimum depth (closest point) at each pixel.
    This is essential for determining which points are visible vs occluded.
    
    Returns:
        depth_buffer: [H, W] minimum depth at each pixel
    """
    H, W = image_shape
    
    # Initialize with maximum depth
    depth_buffer = torch.full((H, W), float('inf'), device=points_cam.device)
    
    # Round 2D coordinates to pixel indices
    pixel_x = points_2d[:, 0].long().clamp(0, W-1)
    pixel_y = points_2d[:, 1].long().clamp(0, H-1)
    
    # Get depths (z-coordinates in camera space)
    depths = torch.abs(points_cam[:, 2])  # Use absolute for positive depths
    
    # For each pixel, keep only the minimum depth (closest point)
    # Use advanced indexing for efficiency
    depth_buffer[pixel_y, pixel_x] = torch.minimum(
        depth_buffer[pixel_y, pixel_x], depths
    )
    
    return depth_buffer


def check_occlusion(
    points_cam: torch.Tensor,          # [N, 3] points in camera coordinates
    points_2d: torch.Tensor,            # [N, 2] projected 2D coordinates
    depth_buffer: torch.Tensor,         # [H, W] minimum depth at each pixel
    occlusion_threshold: float = 0.01  # Tolerance for depth comparison (meters)
) -> torch.Tensor:
    """
    Check which points are occluded by comparing with depth buffer.
    
    A point is considered occluded if its depth is significantly larger than
    the minimum depth stored in the depth buffer at its projected pixel location.
    
    Args:
        occlusion_threshold: Depth tolerance in meters. Points within this
                           distance of the depth buffer value are considered visible.
    
    Returns:
        occlusion_mask: [N] boolean mask, True if point is occluded
    """
    N = len(points_cam)
    H, W = depth_buffer.shape
    
    # Get depths of all points
    point_depths = torch.abs(points_cam[:, 2])
    
    # Round 2D coordinates to pixel indices
    pixel_x = points_2d[:, 0].long().clamp(0, W-1)
    pixel_y = points_2d[:, 1].long().clamp(0, H-1)
    
    # Get minimum depth at each point's pixel location
    min_depths_at_pixels = depth_buffer[pixel_y, pixel_x]
    
    # Point is occluded if its depth is significantly larger than minimum
    occlusion_mask = point_depths > (min_depths_at_pixels + occlusion_threshold)
    
    return occlusion_mask
```

#### Step 2: 2D Segmentation Inference
```python
# Run segmentation on each image
segmentation_results = []
for img in images:
    with torch.no_grad():
        seg_output = segmentation_model_2d(img)
        # seg_output: [H, W, num_classes] - probability distribution
        segmentation_results.append(seg_output)
```

#### Step 3: Back-projection and Accumulation
```python
def accumulate_predictions(
    points_2d: torch.Tensor,        # [N, 2] projected coordinates
    visibility_mask: torch.Tensor,  # [N] boolean mask
    segmentation_2d: torch.Tensor,  # [H, W, C] class probabilities
    vote_matrix: torch.Tensor,      # [N, C] accumulator
    method: str = "majority_vote"
):
    visible_points_2d = points_2d[visibility_mask]
    
    # Bilinear sampling for sub-pixel accuracy
    sampled_probs = bilinear_sample(segmentation_2d, visible_points_2d)
    
    if method == "majority_vote":
        # Add hard predictions
        predicted_classes = torch.argmax(sampled_probs, dim=1)
        vote_matrix[visibility_mask].scatter_add_(
            1, predicted_classes.unsqueeze(1), torch.ones_like(predicted_classes.unsqueeze(1))
        )
    elif method == "confidence_weighted":
        # Add soft probabilities
        vote_matrix[visibility_mask] += sampled_probs
    elif method == "bayesian":
        # Multiplicative update (in log space for stability)
        log_vote_matrix = torch.log(vote_matrix[visibility_mask] + 1e-10)
        log_vote_matrix += torch.log(sampled_probs + 1e-10)
        vote_matrix[visibility_mask] = torch.exp(log_vote_matrix)
```

### 3. Aggregation Strategies

#### Majority Voting
```python
# Simple counting of class predictions
final_labels = torch.argmax(vote_matrix, dim=1)
confidence = vote_matrix.max(dim=1)[0] / vote_matrix.sum(dim=1)
```

#### Confidence-Weighted Averaging
```python
# Weight by prediction confidence
normalized_votes = vote_matrix / vote_matrix.sum(dim=1, keepdim=True)
final_labels = torch.argmax(normalized_votes, dim=1)
confidence = normalized_votes.max(dim=1)[0]
```

#### Uncertainty-Aware Aggregation
```python
# Consider prediction uncertainty from 2D model
def uncertainty_weighted_fusion(
    vote_matrix: torch.Tensor,
    uncertainty_matrix: torch.Tensor  # [N, num_views] uncertainty per view
):
    weights = 1.0 / (uncertainty_matrix + 1e-6)
    weighted_votes = vote_matrix * weights.unsqueeze(-1)
    return weighted_votes.sum(dim=1)
```

## Implementation Considerations

### 1. Occlusion Handling
- **Z-buffering**: Maintain depth buffer to detect occluded points
- **Ray-casting**: Check if point's ray intersects with mesh/surface
- **Multi-scale validation**: Use multiple resolutions to handle edge cases

### 2. View Selection
```python
def select_optimal_views(
    points: torch.Tensor,
    camera_poses: List[torch.Tensor],
    max_views: int = 5
) -> List[int]:
    """Select views that maximize point coverage and minimize redundancy."""
    coverage_scores = []
    for i, pose in enumerate(camera_poses):
        # Compute view frustum
        frustum = compute_frustum(pose, camera_intrinsics)
        
        # Count points in frustum
        in_frustum = check_points_in_frustum(points, frustum)
        
        # Compute view angle diversity
        view_direction = pose[:3, 2]  # Camera forward
        angle_diversity = compute_angle_diversity(view_direction, selected_views)
        
        score = in_frustum.sum() * angle_diversity
        coverage_scores.append(score)
    
    return torch.topk(torch.tensor(coverage_scores), max_views).indices
```

### 3. Temporal Consistency
For sequential data (video/LiDAR sequences):
```python
def temporal_smoothing(
    current_labels: torch.Tensor,
    previous_labels: torch.Tensor,
    point_correspondences: torch.Tensor,  # [N] indices mapping to previous frame
    smoothing_weight: float = 0.3
):
    # Smooth predictions across time
    smoothed_labels = current_labels.clone()
    valid_correspondences = point_correspondences >= 0
    
    previous_probs = F.one_hot(previous_labels[point_correspondences[valid_correspondences]])
    current_probs = F.one_hot(current_labels[valid_correspondences])
    
    smoothed_probs = (1 - smoothing_weight) * current_probs + smoothing_weight * previous_probs
    smoothed_labels[valid_correspondences] = smoothed_probs.argmax(dim=1)
    
    return smoothed_labels
```

## Advantages
1. **Simplicity**: Straightforward to implement with existing projection code
2. **Flexibility**: Works with any 2D segmentation model
3. **Interpretability**: Clear understanding of how labels are assigned
4. **No training required**: Pure geometric approach

## Limitations
1. **View inconsistency**: Different views may predict different labels
2. **Occlusion artifacts**: Hidden points receive no predictions
3. **Boundary bleeding**: Imprecise at object boundaries
4. **Computational cost**: Scales with number of views × number of points

## Performance Optimizations

### 1. Batch Processing
```python
# Process multiple views simultaneously
batched_projections = batch_project_points(
    points.unsqueeze(0).expand(num_views, -1, -1),
    torch.stack(camera_intrinsics),
    torch.stack(camera_extrinsics)
)
```

### 2. Sparse Processing
```python
# Only process points near image boundaries
def process_boundary_points(points, voxel_size=0.1):
    # Downsample for initial prediction
    sparse_points, indices = grid_subsample(points, voxel_size)
    sparse_labels = multiview_fusion(sparse_points, ...)
    
    # Propagate to dense points
    dense_labels = nearest_neighbor_propagation(
        sparse_points, sparse_labels, points
    )
    return dense_labels
```

### 3. GPU Acceleration
```python
# Utilize PyTorch's CUDA operations
with torch.cuda.amp.autocast():
    # Mixed precision for faster inference
    segmentation_output = model(images.cuda())
    
# Custom CUDA kernel for projection
from torch.utils.cpp_extension import load_inline
projection_cuda = load_inline(
    name='projection_cuda',
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=['batch_project_points_cuda']
)
```

## Recommended Libraries and Tools
- **Open3D**: For point cloud visualization and basic operations
- **PyTorch3D**: For differentiable rendering and projection
- **Numba**: For JIT compilation of projection loops
- **CUDA**: For custom high-performance kernels

## Example Usage with Pylon
```python
from utils.point_cloud_ops.rendering import render_rgb, render_segmentation

def pylon_multiview_fusion(
    pc_data: Dict[str, torch.Tensor],
    camera_params: List[Tuple[torch.Tensor, torch.Tensor]],
    images: List[torch.Tensor],
    seg_model_2d
) -> torch.Tensor:
    num_classes = seg_model_2d.num_classes
    vote_matrix = torch.zeros(len(pc_data['pos']), num_classes)
    
    for img, (K, RT) in zip(images, camera_params):
        # Get 2D segmentation
        seg_2d = seg_model_2d(img)
        
        # Use Pylon's rendering to determine visibility
        resolution = (img.shape[-1], img.shape[-2])
        depth_map = render_depth_from_pointcloud(
            pc_data, K, RT, resolution, convention="opengl"
        )
        
        # Back-project and accumulate
        # ... (implementation details)
    
    return torch.argmax(vote_matrix, dim=1)
```