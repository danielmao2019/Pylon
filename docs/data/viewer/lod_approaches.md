# LOD Downsampling Approaches

This document describes four different approaches for intelligent point cloud downsampling that preserve visual quality without expensive screen-space projection.

## Problem Statement

We need to downsample point clouds in 3D space before sending to Plotly, while preserving visual quality. The key challenge is determining which points to keep without doing expensive 3D->2D projection (which would duplicate Plotly's work).

## Approach 1: Distance-Weighted Voxel Sampling

**Core Principle**: Points closer to camera need higher density than points far away.

### Algorithm
```python
def distance_weighted_downsample(points, camera_pos, target_points):
    """Preserve more points closer to camera, fewer points further away."""
    
    # Calculate distance from each point to camera
    distances = torch.norm(points - camera_pos, dim=1)
    
    # Create distance-based voxel sizes
    # Close points: small voxels (high density)
    # Far points: large voxels (low density) 
    max_distance = distances.max()
    normalized_distances = distances / max_distance
    
    # Voxel size scales with distance: close=0.01, far=0.1
    voxel_sizes = 0.01 + 0.09 * normalized_distances**2
    
    # Apply per-point voxel downsampling
    return adaptive_voxel_sample(points, voxel_sizes, target_points)
```

### Advantages
- Natural perceptual model (closer = more detail needed)
- Preserves fine detail where user is focused
- Computationally efficient (only distance calculations)

### Disadvantages  
- May miss important features far from camera
- Doesn't account for surface orientation or importance
- Could create uneven spatial coverage

## Approach 2: Camera-Frustum Spatial Binning

**Core Principle**: Divide 3D space into camera-aligned bins, sample uniformly from each.

### Algorithm
```python
def frustum_based_downsample(points, camera_state, viewport_size, target_points):
    """Divide viewing frustum into bins, sample from each proportionally."""
    
    # Transform points to camera-centered coordinates
    camera_pos = get_camera_position(camera_state)
    forward, right, up = get_camera_vectors(camera_state)
    
    # Project to camera space (cheap - just coordinate transform)
    points_cam = transform_to_camera_space(points, camera_pos, forward, right, up)
    
    # Divide camera space into spatial bins
    # X bins: left-to-right screen regions
    # Y bins: top-to-bottom screen regions  
    # Z bins: near-to-far depth layers
    x_bins = 8   # 8 horizontal regions
    y_bins = 6   # 6 vertical regions
    z_bins = 4   # 4 depth layers
    
    # Assign points to bins based on camera coordinates
    bin_indices = calculate_bin_indices(points_cam, x_bins, y_bins, z_bins)
    
    # Sample from each bin proportionally
    points_per_bin = target_points // (x_bins * y_bins * z_bins)
    
    selected_indices = []
    for bin_id in range(x_bins * y_bins * z_bins):
        bin_points = points[bin_indices == bin_id]
        if len(bin_points) > 0:
            # Sample randomly from this bin
            n_sample = min(points_per_bin, len(bin_points))
            bin_selected = torch.randperm(len(bin_points))[:n_sample]
            selected_indices.extend(bin_points[bin_selected])
    
    return points[selected_indices]
```

### Advantages
- Guarantees spatial coverage across entire view
- Camera-aware sampling (aligned with view direction)
- Predictable point distribution

### Disadvantages
- May oversample empty regions
- Doesn't account for point importance or surface features
- Fixed bin structure may not match point cloud structure

## Approach 3: Angular Importance Sampling

**Core Principle**: Points that subtend larger angles from camera are more visually important.

### Algorithm
```python
def angular_importance_downsample(points, camera_pos, target_points):
    """Sample points based on their angular importance from camera viewpoint."""
    
    # Calculate angular size of each point's local neighborhood
    angular_importance = calculate_angular_importance(points, camera_pos)
    
    # Sample points with probability proportional to angular importance
    probabilities = angular_importance / angular_importance.sum()
    
    # Weighted sampling without replacement
    selected_indices = torch.multinomial(probabilities, target_points, replacement=False)
    
    return points[selected_indices]

def calculate_angular_importance(points, camera_pos):
    """Estimate how much screen space each point will occupy."""
    
    # Distance to camera
    distances = torch.norm(points - camera_pos, dim=1)
    
    # Local point density (how many neighbors nearby)
    local_density = estimate_local_density(points, radius=0.1)
    
    # Angular size ≈ local_area / distance²
    angular_importance = local_density / (distances**2 + 1e-6)
    
    return angular_importance
```

### Advantages
- Theoretically sound (based on actual visual importance)
- Adapts to point cloud structure naturally
- Preserves visually significant regions

### Disadvantages
- Requires expensive local density calculations
- May be unstable for sparse or irregular point clouds
- Complex to tune and debug

## Approach 4: Surface-Aware Sampling

**Core Principle**: Preserve points that represent surface boundaries and edges.

### Algorithm
```python
def surface_aware_downsample(points, camera_state, target_points):
    """Preserve points that represent important surface features."""
    
    # Calculate surface normals and curvature
    normals, curvatures = estimate_surface_properties(points)
    
    # Points with high curvature are more important (edges, corners)
    # Points facing camera are more important than back-facing
    camera_direction = get_camera_forward_vector(camera_state)
    
    # Importance = curvature * visibility
    visibility = torch.clamp(torch.dot(normals, camera_direction), 0, 1)
    importance = curvatures * visibility
    
    # Sample based on importance
    probabilities = importance / importance.sum()
    selected_indices = torch.multinomial(probabilities, target_points, replacement=False)
    
    return points[selected_indices]
```

### Advantages
- Preserves geometric features and surface detail
- Camera-aware (considers viewing angle)
- Maintains shape characteristics

### Disadvantages
- Requires expensive normal and curvature estimation
- May be sensitive to noise in point clouds
- Complex implementation and parameter tuning

## Recommended: Hybrid Approach

**Combines distance-weighting with spatial binning for robust performance.**

### Algorithm
```python
def hybrid_visual_downsample(points, camera_state, viewport_size, target_points):
    """Hybrid approach: distance-weighted + spatial binning."""
    
    # Step 1: Distance-based importance
    camera_pos = get_camera_position(camera_state)
    distances = torch.norm(points - camera_pos, dim=1)
    distance_weights = 1.0 / (distances**2 + 1e-6)  # Closer = more important
    
    # Step 2: Spatial binning to ensure coverage
    bin_indices = spatial_binning(points, camera_state, n_bins=64)
    
    # Step 3: Sample from each bin proportionally to total weight
    selected_indices = []
    for bin_id in range(64):
        bin_mask = (bin_indices == bin_id)
        if bin_mask.sum() > 0:
            bin_points = points[bin_mask]
            bin_weights = distance_weights[bin_mask]
            
            # Points per bin based on total weight in bin
            total_weight = bin_weights.sum()
            points_for_bin = int(target_points * total_weight / distance_weights.sum())
            points_for_bin = min(points_for_bin, len(bin_points))
            
            if points_for_bin > 0:
                # Weighted sampling within bin
                probs = bin_weights / bin_weights.sum()
                bin_selected = torch.multinomial(probs, points_for_bin, replacement=False)
                selected_indices.extend(torch.nonzero(bin_mask)[bin_selected])
    
    return points[torch.cat(selected_indices)]
```

### Advantages
- **Distance weighting**: Ensures close details are preserved
- **Spatial binning**: Ensures good coverage across the view  
- **No expensive projection**: Just geometric calculations
- **Principled**: Based on visual importance, not arbitrary heuristics
- **Robust**: Works well across different point cloud types

### Use Cases
- **Approach 1**: Simple scenes, performance-critical applications
- **Approach 2**: Large, uniform point clouds requiring even coverage
- **Approach 3**: High-quality rendering of complex geometric structures
- **Approach 4**: Scientific visualization requiring feature preservation
- **Hybrid**: General-purpose LOD for interactive point cloud viewer

## Implementation Status

- **Documented**: All four approaches ✅
- **Approved**: Hybrid approach selected for implementation ✅  
- **Status**: Ready for implementation ✅