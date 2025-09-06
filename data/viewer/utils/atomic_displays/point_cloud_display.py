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
from data.transforms.vision_3d.pclod import create_lod_function
from utils.point_cloud_ops.random_select import RandomSelect
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


def _apply_lod_processing(
    pc_dict: Dict[str, torch.Tensor],
    key: Optional[str],
    lod_type: str,
    lod_config: Optional[Dict[str, Any]],
    camera_state: Optional[Dict[str, Any]],
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]],
    point_size: float,
    point_opacity: float,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]],
    original_count: int,
    title: str
) -> Tuple[Dict[str, torch.Tensor], str]:
    """Apply LOD processing to point cloud data.
    
    Args:
        pc_dict: Point cloud dictionary with 'pos' and optional other fields
        key: Optional key name for labels
        lod_type: Type of LOD ("none", "density", "continuous", or "discrete")
        lod_config: LOD configuration dictionary
        camera_state: Optional camera state for syncing views
        point_cloud_id: Unique identifier for LOD caching
        point_size: Size of points for visualization
        point_opacity: Opacity of points for visualization
        axis_ranges: Optional fixed axis ranges
        original_count: Original point count before processing
        title: Current title to update with LOD info
        
    Returns:
        Tuple of (processed_pc_dict, updated_title)
    """
    # Prepare LOD configuration
    effective_lod_config = lod_config or {}
    
    # For advanced LOD (continuous/discrete), we need to add camera_state to config
    if lod_type in ["continuous", "discrete"]:
        if camera_state is None:
            # Create initial figure to let Plotly auto-calculate camera, then extract it
            logger.info(f"Advanced LOD requested with auto-camera - creating initial figure to extract camera state")
            temp_fig = _create_point_cloud_figure(
                pc=pc_dict,
                color_key=key,
                color_type=None,  # No color_type needed for temp figure
                highlight_indices=None,
                point_size=point_size,
                point_opacity=point_opacity,
                axis_ranges=axis_ranges,
                camera_state=None
            )
            effective_camera_state = _extract_default_camera_from_figure(temp_fig, pc_dict['pos'])
        else:
            effective_camera_state = camera_state
        
        # Add camera_state to LOD config
        effective_lod_config = dict(effective_lod_config)  # Make a copy
        effective_lod_config['camera_state'] = effective_camera_state
    
    # Normalize point_cloud_id if provided
    normalized_id = normalize_point_cloud_id(point_cloud_id) if point_cloud_id else None
    
    # Create LOD function using factory
    lod_function = create_lod_function(
        lod_type=lod_type,
        lod_config=effective_lod_config,
        point_cloud_id=normalized_id
    )
    
    # Apply LOD processing
    processed_pc = lod_function(pc_dict)
    
    # Extract processed data
    points_tensor = processed_pc['pos']
    
    # Update title with LOD info
    updated_title = title
    if lod_type == "density" and len(points_tensor) < original_count:
        density_pct = effective_lod_config.get('density', 100)
        density_suffix = f" (Density: {density_pct}%: {len(points_tensor):,}/{original_count:,})"
        updated_title = f"{title}{density_suffix}"
        logger.info(f"Density applied successfully: {original_count} -> {len(points_tensor)} points, title updated")
    elif lod_type in ["continuous", "discrete"] and len(points_tensor) < original_count:
        lod_suffix = f" ({lod_type.title()} LOD: {len(points_tensor):,}/{original_count:,})"
        updated_title = f"{title}{lod_suffix}"
        logger.info(f"LOD applied successfully: {original_count} -> {len(points_tensor)} points, title updated")
    else:
        logger.info(f"No LOD/Density title update: lod_type={lod_type}, original={original_count}, processed={len(points_tensor)}")
    
    # Handle edge case of empty point clouds
    if len(points_tensor) == 0:
        processed_pc['pos'] = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    
    return processed_pc, updated_title


def _apply_browser_downsampling(
    processed_pc: Dict[str, torch.Tensor],
    highlight_indices: Optional[torch.Tensor],
    original_count: int,
    title: str
) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], str]:
    """Apply browser downsampling to point cloud data for memory limits.
    
    Args:
        processed_pc: Point cloud dictionary after LOD processing
        highlight_indices: Optional tensor of indices to highlight
        original_count: Original point count before any processing
        title: Current title to update with downsampling info
        
    Returns:
        Tuple of (downsampled_pc_dict, updated_highlight_indices, updated_title)
    """
    # 🔍 CRITICAL FIX: Downsample for browser memory limits
    # Browser cannot handle 1.6M points - causes "Array buffer allocation failed"
    MAX_BROWSER_POINTS = 50000  # Maximum points browser can handle reliably
    points_tensor = processed_pc['pos']
    
    if len(points_tensor) <= MAX_BROWSER_POINTS:
        return processed_pc, highlight_indices, title
    
    logger.info(f"Applying browser downsampling: {len(points_tensor)} -> {MAX_BROWSER_POINTS} points")
    
    # Use RandomSelect for browser downsampling with deterministic seeding
    # This maintains index chaining and provides reproducible results
    browser_downsample = RandomSelect(count=MAX_BROWSER_POINTS)
    downsampled_pc = browser_downsample(processed_pc, seed=42)  # Fixed seed for reproducibility
    
    # Get final chained indices relative to original point cloud
    # These indices map from final points back to original point cloud
    final_indices = downsampled_pc.get('indices')
    
    # Filter highlight_indices using the final chained indices
    updated_highlight_indices = highlight_indices
    if highlight_indices is not None and final_indices is not None:
        # Create a reverse mapping from original indices to new positions
        # final_indices contains the original indices that were selected
        reverse_mapping = torch.full((original_count,), -1, dtype=torch.int64, device=highlight_indices.device)
        new_positions = torch.arange(len(final_indices), dtype=torch.int64, device=highlight_indices.device)
        reverse_mapping[final_indices] = new_positions
        
        # Map highlight_indices to new positions
        new_highlight_indices = reverse_mapping[highlight_indices]
        
        # Keep only valid indices (those that are >= 0, meaning they were selected)
        valid_mask = new_highlight_indices >= 0
        filtered_highlight_indices = new_highlight_indices[valid_mask]
        
        # Set to None if no valid indices remain
        if filtered_highlight_indices.numel() == 0:
            updated_highlight_indices = None
        else:
            updated_highlight_indices = filtered_highlight_indices
    
    # Update title to show downsampling
    updated_title = f"{title} (Browser limit: {MAX_BROWSER_POINTS:,}/{original_count:,})"
    
    return downsampled_pc, updated_highlight_indices, updated_title


def _create_point_cloud_figure(
    pc: Dict[str, torch.Tensor],
    color_key: Optional[str],
    color_type: Optional[str],
    highlight_indices: Optional[torch.Tensor],
    point_size: float,
    point_opacity: float,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]],
    camera_state: Optional[Dict[str, Any]],
) -> go.Figure:
    """Create base point cloud figure with specified parameters.
    
    Args:
        pc: Point cloud dictionary containing:
            - 'pos': Point cloud positions tensor of shape [N, 3] (required)
            - 'rgb': Optional color tensor of shape [N, 3] or [N, C]
            - Other fields including the label field specified by 'key'
        key: Optional key name to extract labels from pc dictionary (e.g., 'classification', 'labels')
        highlight_indices: Optional tensor of indices to highlight
        point_size: Size of the points in visualization
        point_opacity: Opacity of the points in visualization
        axis_ranges: Optional fixed axis ranges for consistent scaling
        camera_state: Optional camera state for syncing views
        
    Returns:
        Plotly figure for point cloud visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(pc, dict), f"Expected dict for pc, got {type(pc)}"
    assert 'pos' in pc, f"pc must have 'pos' key, got keys: {list(pc.keys())}"
    
    # Extract points from pc dictionary
    points = pc['pos']
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor for pc['pos'], got {type(points)}"
    assert points.ndim == 2, f"Expected 2D tensor [N,3], got shape {points.shape}"
    assert points.shape[1] == 3, f"Expected 3 coordinates, got {points.shape[1]}"
    assert points.numel() > 0, f"Point cloud cannot be empty"
    
    # Check if 'rgb' field is provided first and prepare marker color configuration
    if 'rgb' in pc:
        colors = pc['rgb']
        assert isinstance(colors, torch.Tensor), f"colors must be torch.Tensor, got {type(colors)}"
        assert colors.shape[0] == points.shape[0], f"colors length {colors.shape[0]} != points length {points.shape[0]}"
        # Prepare marker dict for RGB colors
        colors_np = point_cloud_to_numpy(colors)
        marker_color_config = {'color': colors_np}
    else:
        # No 'rgb' field - must use color_key
        assert color_key is not None, f"color_key must be provided when 'rgb' is not in pc dict, got keys: {list(pc.keys())}"
        assert color_key in pc, f"color_key '{color_key}' must be in pc dict, got keys: {list(pc.keys())}"
        
        labels = pc[color_key]
        assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
        assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
        
        # Check if color_type is provided, if not, guess based on color_key
        if color_type is None:
            # Guess color_type based on color_key
            if color_key in ['classification', 'change_map']:
                color_type = 'classification'
            elif color_key == 'density':
                color_type = 'regression'
            else:
                raise ValueError(f"Cannot infer color_type from color_key '{color_key}'. Please specify color_type as 'classification' or 'regression'")
        
        assert isinstance(color_type, str), f"color_type must be str, got {type(color_type)}"
        assert color_type in ["classification", "regression"], f"color_type must be 'classification' or 'regression', got {color_type!r}"
        
        # Prepare marker dict based on color_type
        if color_type == 'classification':
            # Convert labels to colors (in torch)
            colors = _convert_labels_to_colors_torch(labels)
            colors_np = point_cloud_to_numpy(colors)
            marker_color_config = {'color': colors_np}
        else:  # color_type == 'regression'
            # For regression, use Plotly's colorscale with the raw label values
            labels_np = point_cloud_to_numpy(labels)
            # Define global color range for consistent scale
            cmin, cmax = float(labels_np.min()), float(labels_np.max())
            marker_color_config = {
                'color': labels_np,
                'colorscale': 'Viridis',
                'showscale': True,
                'colorbar': dict(title=color_key or 'Value'),
                'cmin': cmin,
                'cmax': cmax
            }
    
    # Input validation for highlight_indices
    if highlight_indices is not None:
        assert isinstance(highlight_indices, torch.Tensor), f"highlight_indices must be torch.Tensor, got {type(highlight_indices)}"
        assert highlight_indices.dtype in [torch.int32, torch.int64], f"highlight_indices must be integer tensor, got {highlight_indices.dtype}"
        assert torch.all(highlight_indices >= 0), f"highlight_indices must be non-negative, got min: {highlight_indices.min()}"
        assert torch.all(highlight_indices < len(points)), f"highlight_indices must be < len(points)={len(points)}, got max: {highlight_indices.max()}"
    
    # Convert points to numpy for Plotly
    points_np = point_cloud_to_numpy(points)
    
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
            # Prepare marker config for non-highlighted points
            marker_config = dict(size=point_size, opacity=point_opacity * 0.05)  # Decreased opacity
            
            # Apply color configuration with masking
            assert 'color' in marker_color_config, f"marker_color_config must have 'color' key, got keys: {list(marker_color_config.keys())}"
            marker_config.update(marker_color_config)
            marker_config['color'] = marker_color_config['color'][non_highlight_mask]
            
            non_highlight_kwargs = dict(
                x=points_np[non_highlight_mask, 0],
                y=points_np[non_highlight_mask, 1],
                z=points_np[non_highlight_mask, 2],
                mode='markers',
                marker=marker_config,
                hoverinfo='skip',
                showlegend=False
            )
            
            fig.add_trace(go.Scatter3d(**non_highlight_kwargs))
        
        if highlight_mask.any():  # Add highlighted points trace
            # Prepare marker config for highlighted points
            marker_config = dict(size=point_size, opacity=point_opacity)  # Full opacity
            
            # Apply color configuration with masking
            assert 'color' in marker_color_config, f"marker_color_config must have 'color' key, got keys: {list(marker_color_config.keys())}"
            marker_config.update(marker_color_config)
            marker_config['color'] = marker_color_config['color'][highlight_mask]
            marker_config['showscale'] = False  # Don't show scale twice
            
            highlight_kwargs = dict(
                x=points_np[highlight_mask, 0],
                y=points_np[highlight_mask, 1],
                z=points_np[highlight_mask, 2],
                mode='markers',
                marker=marker_config,
                hoverinfo='skip',
                showlegend=False
            )
            
            fig.add_trace(go.Scatter3d(**highlight_kwargs))
    else:
        # No highlighting - single trace with uniform opacity
        marker_config = dict(size=point_size, opacity=point_opacity)
        marker_config.update(marker_color_config)
        
        scatter3d_kwargs = dict(
            x=points_np[:, 0],
            y=points_np[:, 1],
            z=points_np[:, 2],
            mode='markers',
            marker=marker_config,
            hoverinfo='skip',
            showlegend=False
        )
        
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


def create_point_cloud_display(
    pc: Dict[str, torch.Tensor],
    title: str,
    color_key: Optional[str] = None,
    color_type: Optional[str] = None,
    highlight_indices: Optional[torch.Tensor] = None,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "none",
    lod_config: Optional[Dict[str, Any]] = None,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs: Any
) -> go.Figure:
    """Create point cloud display with LOD optimization.
    
    Args:
        pc: Point cloud dictionary containing:
            - 'pos': Point cloud positions tensor of shape [N, 3] (required)
            - 'rgb': Optional color tensor of shape [N, 3] or [N, C]
            - Other fields including the label field specified by 'key'
        key: Optional key name to extract labels from pc dictionary (e.g., 'classification', 'labels')
        highlight_indices: Optional tensor of indices to highlight
        title: Title for the point cloud display
        point_size: Size of the points in visualization
        point_opacity: Opacity of the points in visualization
        camera_state: Optional camera state for syncing views (None for auto-calculation)
        lod_type: Type of LOD ("none", "density", "continuous", or "discrete")
        lod_config: LOD configuration dictionary:
            - For "none": Should be None or empty dict
            - For "density": {"density": int} where density is percentage (1-100)
            - For "continuous": {"camera_state": dict, ...other params...}
            - For "discrete": {"camera_state": dict, ...other params...}
        point_cloud_id: Unique identifier for LOD caching
        axis_ranges: Optional fixed axis ranges for consistent scaling
        **kwargs: Additional arguments
        
    Returns:
        Plotly figure for point cloud visualization
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(pc, dict), f"Expected dict for pc, got {type(pc)}"
    assert 'pos' in pc, f"pc must have 'pos' key, got keys: {list(pc.keys())}"
    
    # Extract points, colors, and labels from pc dictionary
    points = pc['pos']
    colors = pc.get('rgb', None)
    labels = pc.get(color_key, None) if color_key is not None else None
    
    # Validate extracted tensors
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor for pc['pos'], got {type(points)}"
    
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
    
    if axis_ranges is not None:
        assert isinstance(axis_ranges, dict), f"axis_ranges must be dict, got {type(axis_ranges)}"
    
    original_count = len(points)
    
    # Prepare point cloud dictionary
    pc_dict = {'pos': points.float()}  # Ensure float type for computations
    if colors is not None:
        pc_dict['rgb'] = colors
    if labels is not None:
        pc_dict['labels'] = labels
    
    # Apply LOD processing
    processed_pc, title = _apply_lod_processing(
        pc_dict=pc_dict,
        key=color_key,
        lod_type=lod_type,
        lod_config=lod_config,
        camera_state=camera_state,
        point_cloud_id=point_cloud_id,
        point_size=point_size,
        point_opacity=point_opacity,
        axis_ranges=axis_ranges,
        original_count=original_count,
        title=title
    )
    
    # Apply browser downsampling for memory limits
    processed_pc, highlight_indices, title = _apply_browser_downsampling(
        processed_pc=processed_pc,
        highlight_indices=highlight_indices,
        original_count=original_count,
        title=title
    )
    
    # Create final figure with LOD-processed data
    fig = _create_point_cloud_figure(
        pc=processed_pc,
        color_key=color_key,
        color_type=color_type,
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
