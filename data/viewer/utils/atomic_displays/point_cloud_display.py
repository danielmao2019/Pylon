"""Point cloud display utilities for 3D point cloud visualization.

API Design Principles:
- torch.Tensor: Used for ALL computational operations (LOD processing, distance calculations)
- numpy.ndarray: Used ONLY for final Plotly visualization (at the very end)
- Clear boundaries: Both APIs only accept torch.Tensor inputs
- Fail fast: Use assertions to enforce API contracts

Data Flow:
1. Dataset -> torch.Tensor (CPU)
2. apply_lod_to_point_cloud -> torch.Tensor ONLY (enforced by assertions)
3. create_point_cloud_figure -> torch.Tensor ONLY (enforced by assertions)
4. Plotly visualization -> numpy conversion (internal, at the very end)
"""
from typing import Dict, Optional, Union, Any, Tuple
import numpy as np
import torch
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color
from data.viewer.utils.continuous_lod import ContinuousLOD
from data.viewer.utils.discrete_lod import DiscreteLOD
import logging

logger = logging.getLogger(__name__)


def build_point_cloud_id(datapoint: Dict[str, Any], component: str) -> Tuple[str, int, str]:
    """Build structured point cloud ID from datapoint context.
    
    Args:
        datapoint: Contains meta_info with idx; dataset info from backend
        component: Point cloud component (source, target, pc_1, pc_2, change_map, etc.)
        
    Returns:
        Tuple of (dataset_name, datapoint_idx, component)
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
    assert isinstance(component, str), f"component must be str, got {type(component)}"
    
    from data.viewer.callbacks import registry  # Import here to avoid circular imports
    
    meta_info = datapoint.get('meta_info', {})
    datapoint_idx = meta_info.get('idx', 0)
    
    # Get dataset name from backend (handles case where app isn't fully initialized)
    dataset_name = 'unknown'
    if hasattr(registry, 'viewer') and hasattr(registry.viewer, 'backend'):
        dataset_name = getattr(registry.viewer.backend, 'current_dataset', 'unknown')
    
    return (dataset_name, datapoint_idx, component)


def normalize_point_cloud_id(point_cloud_id: Union[str, Tuple[str, ...]]) -> str:
    """Normalize point cloud ID to string cache key.
    
    Args:
        point_cloud_id: Either string or tuple (dataset, datapoint_idx, component)
        
    Returns:
        Normalized string cache key
        
    Examples:
        "simple_id" -> "simple_id"
        ("pcr/kitti", 42, "source") -> "pcr/kitti:42:source"
        ("change_detection", 10, "union") -> "change_detection:10:union"
    """
    if isinstance(point_cloud_id, str):
        return point_cloud_id
    else:
        # Convert tuple to colon-separated string
        return ":".join(str(part) for part in point_cloud_id)


def point_cloud_to_numpy(points: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a numpy array.
    
    Args:
        points: Point cloud data as tensor or array
        
    Returns:
        Numpy array representation of the data
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(points, (torch.Tensor, np.ndarray)), f"Expected torch.Tensor or np.ndarray, got {type(points)}"
    
    if isinstance(points, torch.Tensor):
        return points.cpu().numpy()
    return points


def _convert_labels_to_colors_torch(labels: torch.Tensor) -> torch.Tensor:
    """Convert integer labels to RGB colors using torch tensors.
    
    Args:
        labels: Integer label tensor of shape (N,)
        
    Returns:
        RGB color tensor of shape (N, 3) with values in [0, 255]
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
    assert labels.ndim == 1, f"labels must be 1D, got shape {labels.shape}"
    
    device = labels.device
    unique_labels = torch.unique(labels)
    
    # Create color mapping
    colors = torch.zeros((len(labels), 3), dtype=torch.uint8, device=device)
    
    for label in unique_labels:
        # Get color string from segmentation utility
        color_hex = get_color(label.item())
        
        # Convert hex to RGB
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        
        # Apply to matching labels
        mask = (labels == label)
        colors[mask, :] = torch.tensor([r, g, b], dtype=torch.uint8, device=device)
    
    return colors


def apply_lod_to_point_cloud(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: Optional[str] = None,
    lod_config: Optional[Dict[str, Any]] = None,
    density_percentage: Optional[int] = None,
    point_cloud_id: Optional[Union[str, Tuple[str, ...]]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Apply Level of Detail processing to point cloud data.
    
    This function works entirely with torch tensors for optimal performance.
    All inputs must be torch tensors - use numpy->torch conversion before calling.
    
    Args:
        points: Point cloud positions as torch.Tensor (N, 3)
        colors: Optional color data as torch.Tensor (N, 3) or (N, C)
        labels: Optional label data as torch.Tensor (N,)
        camera_state: Camera viewing state dictionary (required for continuous/discrete LOD)
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        lod_config: Optional LOD configuration parameters
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)
        point_cloud_id: Unique identifier for discrete LOD caching
        
    Returns:
        Tuple of (processed_points, processed_colors, processed_labels) as torch tensors
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    
    logger.info(f"apply_lod_to_point_cloud called: points={points.shape}, lod_type={lod_type}, density_percentage={density_percentage}")
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    
    if colors is not None:
        assert isinstance(colors, torch.Tensor), f"colors must be torch.Tensor, got {type(colors)}"
        assert colors.shape[0] == points.shape[0], f"colors length {colors.shape[0]} != points length {points.shape[0]}"
    
    if labels is not None:
        assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
        assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
    
    # Ensure float type for computations
    points = points.float()
    
    # Early return if no processing needed
    if lod_type is None or (lod_type == "none" and (density_percentage is None or density_percentage >= 100)):
        logger.info(f"No processing applied: lod_type={lod_type}, density_percentage={density_percentage}")
        return points, colors, labels
    
    # Prepare point cloud dictionary
    pc_dict = {'pos': points}
    if colors is not None:
        pc_dict['rgb'] = colors
    if labels is not None:
        pc_dict['labels'] = labels
    
    # Apply processing based on type
    if lod_type == "none":
        # Validate density_percentage with assertions
        assert density_percentage is not None, "density_percentage is required when lod_type is 'none'"
        assert isinstance(density_percentage, int), f"density_percentage must be int, got {type(density_percentage)}"
        assert 1 <= density_percentage <= 100, f"density_percentage must be 1-100, got {density_percentage}"
        assert density_percentage < 100, f"density_percentage < 100 should be handled by early return, got {density_percentage}"
        
        # Density-based subsampling
        assert point_cloud_id is not None, "point_cloud_id is required for density-based subsampling"
        from data.viewer.utils.density_lod import DensityLOD
        density_lod = DensityLOD()
        normalized_id = normalize_point_cloud_id(point_cloud_id)
        downsampled = density_lod.subsample(normalized_id, density_percentage, pc_dict)
        logger.info(f"Density subsampling: {len(points)} -> {len(downsampled['pos'])} points ({density_percentage}%)")
        
    elif lod_type == "continuous":
        # Continuous LOD requires camera state - fail fast if not provided
        assert camera_state is not None, "camera_state is required for continuous LOD"
        lod = ContinuousLOD(**(lod_config or {}))
        downsampled = lod.subsample(pc_dict, camera_state)
        logger.info(f"Continuous LOD applied: {len(points)} -> {len(downsampled['pos'])} points")
        
    elif lod_type == "discrete":
        # Discrete LOD requires camera state and point cloud ID - fail fast if not provided
        assert camera_state is not None, "camera_state is required for discrete LOD"
        assert point_cloud_id is not None, "point_cloud_id is required for discrete LOD"
        lod = DiscreteLOD(**(lod_config or {}))
        normalized_id = normalize_point_cloud_id(point_cloud_id)
        downsampled = lod.subsample(normalized_id, camera_state, pc_dict)
        logger.info(f"Discrete LOD applied: {len(points)} -> {len(downsampled['pos'])} points")
        
    else:
        raise ValueError(f"Unknown LOD type: {lod_type}. Must be 'none', 'continuous', or 'discrete'.")
        
    # Extract processed data
    processed_points = downsampled['pos']
    processed_colors = downsampled.get('rgb')
    processed_labels = downsampled.get('labels')
    
    return processed_points, processed_colors, processed_labels


def _create_base_point_cloud_figure(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
    highlight_indices: Optional[torch.Tensor],
    point_size: float,
    point_opacity: float,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]],
    camera_state: Optional[Dict[str, Any]],
) -> go.Figure:
    """Create base point cloud figure with specified parameters."""
    # Input validation for highlight_indices
    if highlight_indices is not None:
        assert isinstance(highlight_indices, torch.Tensor), f"highlight_indices must be torch.Tensor, got {type(highlight_indices)}"
        assert highlight_indices.dtype in [torch.int32, torch.int64], f"highlight_indices must be integer tensor, got {highlight_indices.dtype}"
        assert torch.all(highlight_indices >= 0), f"highlight_indices must be non-negative, got min: {highlight_indices.min()}"
        assert torch.all(highlight_indices < len(points)), f"highlight_indices must be < len(points)={len(points)}, got max: {highlight_indices.max()}"
    
    # Process colors from labels if needed (keep in torch tensors)
    if colors is not None:
        final_colors = colors
    elif labels is not None:
        # Convert labels to colors (in torch)
        final_colors = _convert_labels_to_colors_torch(labels)
    else:
        final_colors = None
    
    # Convert to numpy ONLY for Plotly (at the absolute final moment)
    points_np = point_cloud_to_numpy(points)
    colors_np = point_cloud_to_numpy(final_colors) if final_colors is not None else None
    
    # Create figure and add traces based on highlighting logic
    fig = go.Figure()
    
    if highlight_indices is not None:
        highlight_indices_np = highlight_indices.cpu().numpy()
        
        # Create mask for all points
        all_indices = np.arange(len(points))
        highlight_mask = np.isin(all_indices, highlight_indices_np)
        non_highlight_mask = ~highlight_mask
        
        # Create separate traces for highlighted and non-highlighted points
        if non_highlight_mask.any():  # Add non-highlighted points trace
            non_highlight_kwargs = dict(
                x=points_np[non_highlight_mask, 0],
                y=points_np[non_highlight_mask, 1],
                z=points_np[non_highlight_mask, 2],
                mode='markers',
                marker=dict(size=point_size, opacity=point_opacity * 0.05),  # Decreased opacity
                hoverinfo='skip',
                showlegend=False
            )
            
            if colors_np is not None:
                non_highlight_kwargs['marker']['color'] = colors_np[non_highlight_mask]
            else:
                non_highlight_kwargs['marker']['color'] = 'steelblue'
            
            fig.add_trace(go.Scatter3d(**non_highlight_kwargs))
        
        if highlight_mask.any():  # Add highlighted points trace
            highlight_kwargs = dict(
                x=points_np[highlight_mask, 0],
                y=points_np[highlight_mask, 1],
                z=points_np[highlight_mask, 2],
                mode='markers',
                marker=dict(size=point_size, opacity=point_opacity),  # Full opacity
                hoverinfo='skip',
                showlegend=False
            )
            
            if colors_np is not None:
                highlight_kwargs['marker']['color'] = colors_np[highlight_mask]
            else:
                highlight_kwargs['marker']['color'] = 'red'  # Different color for highlights
            
            fig.add_trace(go.Scatter3d(**highlight_kwargs))
    else:
        # No highlighting - single trace with uniform opacity
        scatter3d_kwargs = dict(
            x=points_np[:, 0],
            y=points_np[:, 1],
            z=points_np[:, 2],
            mode='markers',
            marker=dict(size=point_size, opacity=point_opacity),
            hoverinfo='skip',
            showlegend=False
        )
        
        if colors_np is not None:
            scatter3d_kwargs['marker']['color'] = colors_np
        else:
            scatter3d_kwargs['marker']['color'] = 'steelblue'
        
        fig.add_trace(go.Scatter3d(**scatter3d_kwargs))
    
    # Calculate bounding box - use provided ranges if available, otherwise compute from data
    if axis_ranges:
        x_range = list(axis_ranges.get('x', [points_np[:, 0].min(), points_np[:, 0].max()]))
        y_range = list(axis_ranges.get('y', [points_np[:, 1].min(), points_np[:, 1].max()]))
        z_range = list(axis_ranges.get('z', [points_np[:, 2].min(), points_np[:, 2].max()]))
    else:
        x_range = [points_np[:, 0].min(), points_np[:, 0].max()]
        y_range = [points_np[:, 1].min(), points_np[:, 1].max()]
        z_range = [points_np[:, 2].min(), points_np[:, 2].max()]
    
    # Set layout - use provided camera for syncing, or let Plotly auto-calculate
    scene_dict = dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data',  # Use data aspect mode for proper point cloud scaling
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range),
        zaxis=dict(range=z_range)
    )
    
    # Only apply camera if provided (for syncing views)
    # If None, Plotly will auto-calculate appropriate view
    if camera_state is not None:
        scene_dict['camera'] = camera_state
    
    fig.update_layout(scene=scene_dict)
    return fig


def _extract_default_camera_from_figure(fig: go.Figure, points: torch.Tensor) -> Dict[str, Any]:
    """Extract Plotly's auto-calculated camera state from a figure.
    
    When camera_state=None, Plotly automatically calculates optimal camera positioning.
    This function extracts that calculated camera state so it can be used for LOD processing.
    """
    # Get point cloud bounds
    points_np = points.cpu().numpy()
    pc_center = points_np.mean(axis=0)
    pc_size = points_np.max(axis=0) - points_np.min(axis=0)
    max_dim = np.max(pc_size)
    
    # Plotly's default camera positioning logic (approximation)
    # When no camera is specified, Plotly typically positions camera at:
    # - Distance: ~1.5-2x the data span from center
    # - Angle: Diagonal view (positive x, y, z from center)
    distance_factor = 1.5
    camera_offset = max_dim * distance_factor
    
    # Calculate camera position (Plotly's typical auto-calculation)
    eye_pos = pc_center + np.array([camera_offset, camera_offset, camera_offset * 0.7])
    
    # Create camera state in Plotly format
    camera_state = {
        'eye': {'x': float(eye_pos[0]), 'y': float(eye_pos[1]), 'z': float(eye_pos[2])},
        'center': {'x': float(pc_center[0]), 'y': float(pc_center[1]), 'z': float(pc_center[2])},
        'up': {'x': 0, 'y': 0, 'z': 1}
    }
    
    logger.info(f"Extracted auto-calculated camera state: eye={camera_state['eye']}, center={camera_state['center']}")
    return camera_state


def create_point_cloud_display(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    highlight_indices: Optional[torch.Tensor] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "none",
    lod_config: Optional[Dict[str, Any]] = None,
    density_percentage: Optional[int] = None,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs: Any
) -> go.Figure:
    """Create point cloud display with LOD optimization.
    
    This function works with torch tensors throughout processing and only converts
    to numpy arrays at the very end for Plotly visualization.
    
    Args:
        points: Point cloud positions tensor of shape [N, 3]
        colors: Optional color tensor of shape [N, 3] or [N, C]
        labels: Optional label tensor of shape [N]
        title: Title for the point cloud display
        point_size: Size of the points in visualization
        point_opacity: Opacity of the points in visualization
        camera_state: Optional camera state for syncing views (None for auto-calculation)
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        lod_config: Optional LOD configuration parameters
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)
        point_cloud_id: Unique identifier for LOD caching
        axis_ranges: Optional fixed axis ranges for consistent scaling
        **kwargs: Additional arguments
        
    Returns:
        Plotly figure for point cloud visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions (validate points first for logging)
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"
    
    logger.info(f"create_point_cloud_display called: points={points.shape}, lod_type={lod_type}, point_cloud_id={point_cloud_id}")
    assert points.ndim == 2, f"Expected 2D tensor [N,3], got shape {points.shape}"
    assert points.shape[1] == 3, f"Expected 3 coordinates, got {points.shape[1]}"
    assert points.numel() > 0, f"Point cloud cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    assert isinstance(point_size, (int, float)), f"Expected numeric point_size, got {type(point_size)}"
    assert isinstance(point_opacity, (int, float)), f"Expected numeric point_opacity, got {type(point_opacity)}"
    assert 0.0 <= point_opacity <= 1.0, f"point_opacity must be in [0,1], got {point_opacity}"
    
    if colors is not None:
        assert isinstance(colors, torch.Tensor), f"colors must be torch.Tensor, got {type(colors)}"
        assert colors.shape[0] == points.shape[0], f"colors length {colors.shape[0]} != points length {points.shape[0]}"
    
    if labels is not None:
        assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
        assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
    
    if camera_state is not None:
        assert isinstance(camera_state, dict), f"camera_state must be dict, got {type(camera_state)}"
    
    if lod_config is not None:
        assert isinstance(lod_config, dict), f"lod_config must be dict, got {type(lod_config)}"
    
    if density_percentage is not None:
        assert isinstance(density_percentage, int), f"density_percentage must be int, got {type(density_percentage)}"
        assert 1 <= density_percentage <= 100, f"density_percentage must be 1-100, got {density_percentage}"
    
    if axis_ranges is not None:
        assert isinstance(axis_ranges, dict), f"axis_ranges must be dict, got {type(axis_ranges)}"
    
    original_count = len(points)
    
    # For advanced LOD (continuous/discrete), we need to determine effective camera state
    if lod_type in ["continuous", "discrete"]:
        if camera_state is None:
            # Create initial figure to let Plotly auto-calculate camera, then extract it
            logger.info(f"Advanced LOD requested with auto-camera - creating initial figure to extract camera state")
            temp_fig = _create_base_point_cloud_figure(
                points=points,
                colors=colors,
                labels=labels,
                highlight_indices=None,
                point_size=point_size,
                point_opacity=point_opacity,
                axis_ranges=axis_ranges,
                camera_state=None
            )
            effective_camera_state = _extract_default_camera_from_figure(temp_fig, points)
        else:
            effective_camera_state = camera_state
    else:
        effective_camera_state = camera_state  # May be None for simple LOD
    
    # Apply LOD processing (pure torch tensor operations)
    points_tensor, colors_tensor, labels_tensor = apply_lod_to_point_cloud(
        points=points,
        colors=colors,
        labels=labels,
        camera_state=effective_camera_state,  # Use effective camera state
        lod_type=lod_type,
        lod_config=lod_config,
        density_percentage=density_percentage,
        point_cloud_id=point_cloud_id
    )
    
    # Update title with LOD info
    if lod_type == "none" and density_percentage is not None and density_percentage < 100 and len(points_tensor) < original_count:
        density_suffix = f" (Density: {density_percentage}%: {len(points_tensor):,}/{original_count:,})"
        title = f"{title}{density_suffix}"
        logger.info(f"Density applied successfully: {original_count} -> {len(points_tensor)} points, title updated")
    elif lod_type and lod_type != "none" and len(points_tensor) < original_count:
        lod_suffix = f" ({lod_type.title()} LOD: {len(points_tensor):,}/{original_count:,})"
        title = f"{title}{lod_suffix}"
        logger.info(f"LOD applied successfully: {original_count} -> {len(points_tensor)} points, title updated")
    else:
        logger.info(f"No LOD/Density title update: lod_type={lod_type}, density_percentage={density_percentage}, original={original_count}, processed={len(points_tensor)}")
    
    # Handle edge case of empty point clouds
    if len(points_tensor) == 0:
        points_tensor = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    
    # 🔍 CRITICAL FIX: Downsample for browser memory limits
    # Browser cannot handle 1.6M points - causes "Array buffer allocation failed"
    MAX_BROWSER_POINTS = 50000  # Maximum points browser can handle reliably
    if len(points_tensor) > MAX_BROWSER_POINTS:
        print(f"⚠️ [POINT_CLOUD_DISPLAY] Downsampling {len(points_tensor)} points to {MAX_BROWSER_POINTS} for browser rendering")
        # Store original size before downsampling
        original_size = len(points_tensor)
        
        # Random sampling to preserve overall structure
        sample_indices = torch.randperm(original_size)[:MAX_BROWSER_POINTS]
        points_tensor = points_tensor[sample_indices]
        if colors_tensor is not None:
            colors_tensor = colors_tensor[sample_indices]
        if labels_tensor is not None:
            labels_tensor = labels_tensor[sample_indices]
        
        # Filter highlight_indices to only include indices that are in the sampled subset
        if highlight_indices is not None:
            # Create a reverse mapping tensor for fast lookup on the same device as highlight_indices
            # Size must be the original point cloud size since sample_indices contains values up to original_size-1
            reverse_mapping = torch.full((original_size,), -1, dtype=torch.int64, device=highlight_indices.device)
            # Set the new indices for sampled points (ensure arange is on same device too)
            reverse_mapping[sample_indices] = torch.arange(len(sample_indices), dtype=torch.int64, device=highlight_indices.device)
            
            # Map highlight_indices to new indices using vectorized lookup
            # This will be -1 for indices not in sample
            new_highlight_indices = reverse_mapping[highlight_indices]
            
            # Keep only valid indices (those that are >= 0)
            valid_mask = new_highlight_indices >= 0
            filtered_highlight_indices = new_highlight_indices[valid_mask]
            
            # Set to None if no valid indices remain
            if filtered_highlight_indices.numel() == 0:
                highlight_indices = None
            else:
                highlight_indices = filtered_highlight_indices
        
        # Update title to show downsampling
        title = f"{title} (Downsampled: {MAX_BROWSER_POINTS:,}/{original_count:,})"
    
    # Create final figure with LOD-processed data
    fig = _create_base_point_cloud_figure(
        points=points_tensor,
        colors=colors_tensor,
        labels=labels_tensor,
        highlight_indices=highlight_indices,
        point_size=point_size,
        point_opacity=point_opacity,
        axis_ranges=axis_ranges,
        camera_state=camera_state
    )
    
    fig.update_layout(
        title=title,
        uirevision='camera',  # This ensures camera views stay in sync
        margin=dict(l=0, r=40, b=0, t=40),
        height=500,
    )
    
    return fig


def get_point_cloud_display_stats(
    pc_dict: Dict[str, torch.Tensor],
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> Dict[str, Any]:
    """Get point cloud statistics for display.
    
    Args:
        pc_dict: Point cloud dictionary with required 'pos' key and optional other fields
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names
        
    Returns:
        Dictionary containing point cloud statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(pc_dict, dict), f"Expected dict, got {type(pc_dict)}"
    assert 'pos' in pc_dict, f"pc_dict must have 'pos' key, got keys: {list(pc_dict.keys())}"
    
    points = pc_dict['pos']
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"Expected 2D tensor [N,D], got shape {points.shape}"
    assert points.shape[1] >= 3, f"Expected at least 3 coordinates, got {points.shape[1]}"
    assert points.numel() > 0, f"Point cloud cannot be empty"
    
    if change_map is not None:
        assert isinstance(change_map, torch.Tensor), f"change_map must be torch.Tensor, got {type(change_map)}"
        assert change_map.shape[0] == points.shape[0], f"change_map length {change_map.shape[0]} != points length {points.shape[0]}"
    
    if class_names is not None:
        assert isinstance(class_names, dict), f"class_names must be dict, got {type(class_names)}"
    
    # Convert to numpy for stats computation
    points_np = points.detach().cpu().numpy()
    
    stats = {
        'available_fields': list(pc_dict.keys()),
        'total_points': len(points_np),
        'dimensions': points_np.shape[1],
        'x_range': [float(points_np[:, 0].min()), float(points_np[:, 0].max())],
        'y_range': [float(points_np[:, 1].min()), float(points_np[:, 1].max())],
        'z_range': [float(points_np[:, 2].min()), float(points_np[:, 2].max())],
        'center': [float(points_np[:, 0].mean()), float(points_np[:, 1].mean()), float(points_np[:, 2].mean())]
    }
    
    # Add class distribution if change_map is provided
    if change_map is not None:
        unique_classes, class_counts = torch.unique(change_map, return_counts=True)
        unique_classes = unique_classes.cpu().numpy()
        class_counts = class_counts.cpu().numpy()
        total_points = change_map.numel()
        
        class_distribution = {}
        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / total_points) * 100
            cls_key = cls.item() if hasattr(cls, 'item') else cls
            class_name = class_names[cls_key] if class_names and cls_key in class_names else f"Class {cls_key}"
            class_distribution[class_name] = {
                'count': int(count),
                'percentage': float(percentage)
            }
        
        stats['class_distribution'] = class_distribution
    
    return stats
