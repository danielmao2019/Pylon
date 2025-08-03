"""Point cloud display utilities for 3D point cloud visualization.

API Design Principles:
- torch.Tensor: Used for ALL computational operations (LOD processing, distance calculations)
- numpy.ndarray: Used ONLY for final Plotly visualization (at the very end)
- Clear boundaries: Both APIs only accept torch.Tensor inputs
- Fail fast: Use assertions to enforce API contracts

Data Flow:
1. Dataset → torch.Tensor (CPU)
2. apply_lod_to_point_cloud → torch.Tensor ONLY (enforced by assertions)
3. create_point_cloud_figure → torch.Tensor ONLY (enforced by assertions)
4. Plotly visualization → numpy conversion (internal, at the very end)
"""
from typing import Dict, Optional, Union, Any, Tuple
import numpy as np
import torch
from dash import html
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
    
    # Get dataset name from backend
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
    """
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
    All inputs must be torch tensors - use numpy→torch conversion before calling.
    
    Args:
        points: Point cloud positions as torch.Tensor (N, 3)
        colors: Optional color data as torch.Tensor (N, 3) or (N, C)
        labels: Optional label data as torch.Tensor (N,)
        camera_state: Camera viewing state dictionary
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        lod_config: Optional LOD configuration parameters
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)
        point_cloud_id: Unique identifier for discrete LOD caching
        
    Returns:
        Tuple of (processed_points, processed_colors, processed_labels) as torch tensors
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    logger.info(f"apply_lod_to_point_cloud called: points={points.shape}, lod_type={lod_type}, density_percentage={density_percentage}")
    
    # Input validation
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
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
        # Continuous LOD processing
        assert camera_state is not None, "camera_state is required for continuous LOD"
        lod = ContinuousLOD(**(lod_config or {}))
        downsampled = lod.subsample(pc_dict, camera_state)
        logger.info(f"Continuous LOD applied: {len(points)} -> {len(downsampled['pos'])} points")
        
    elif lod_type == "discrete":
        # Discrete LOD processing
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


def create_point_cloud_figure(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: Optional[str] = "continuous",
    lod_config: Optional[Dict[str, Any]] = None,
    density_percentage: Optional[int] = None,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> go.Figure:
    """Create a 3D point cloud visualization figure with optional LOD.

    This function works with torch tensors throughout processing and only converts
    to numpy arrays at the very end for Plotly visualization.

    Args:
        points: Point cloud positions as torch.Tensor (N, 3)
        colors: Optional color data as torch.Tensor (N, 3) or (N, C)
        labels: Optional label data as torch.Tensor (N,)
        title: Title for the figure
        point_size: Size of the points
        point_opacity: Opacity of the points
        camera_state: Optional dictionary containing camera position state
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        lod_config: Optional LOD configuration parameters
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)
        point_cloud_id: Unique identifier for discrete LOD caching

    Returns:
        Plotly Figure object with potentially downsampled point cloud
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    logger.info(f"create_point_cloud_figure called: points={points.shape}, lod_type={lod_type}, point_cloud_id={point_cloud_id}")
    
    # Input validation
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2 and points.shape[1] == 3, f"points must be (N, 3), got {points.shape}"
    
    if colors is not None:
        assert isinstance(colors, torch.Tensor), f"colors must be torch.Tensor, got {type(colors)}"
        assert colors.shape[0] == points.shape[0], f"colors length {colors.shape[0]} != points length {points.shape[0]}"
    
    if labels is not None:
        assert isinstance(labels, torch.Tensor), f"labels must be torch.Tensor, got {type(labels)}"
        assert labels.shape[0] == points.shape[0], f"labels length {labels.shape[0]} != points length {points.shape[0]}"
    
    original_count = len(points)
    
    # Apply LOD processing (pure torch tensor operations)
    points_tensor, colors_tensor, labels_tensor = apply_lod_to_point_cloud(
        points=points,
        colors=colors,
        labels=labels,
        camera_state=camera_state,
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
    
    # Process colors from labels if needed (keep in torch tensors)
    if colors_tensor is not None:
        assert colors_tensor.shape[0] == points_tensor.shape[0], f"colors length {colors_tensor.shape[0]} != points length {points_tensor.shape[0]}"
        final_colors = colors_tensor
    elif labels_tensor is not None:
        assert labels_tensor.shape[0] == points_tensor.shape[0], f"labels length {labels_tensor.shape[0]} != points length {points_tensor.shape[0]}"
        # Convert labels to colors (in torch)
        final_colors = _convert_labels_to_colors_torch(labels_tensor)
    else:
        final_colors = None
    
    # Convert to numpy ONLY for Plotly (at the absolute final moment)
    points_np = point_cloud_to_numpy(points_tensor)
    colors_np = point_cloud_to_numpy(final_colors) if final_colors is not None else None
    
    # Create Plotly scatter plot 
    scatter3d_kwargs = dict(
        x=points_np[:, 0],
        y=points_np[:, 1],
        z=points_np[:, 2],
        mode='markers',
        marker=dict(size=point_size, opacity=point_opacity),
        hoverinfo='skip',  # Disable hover for performance
    )
    
    if colors_np is not None:
        scatter3d_kwargs['marker']['color'] = colors_np
    else:
        scatter3d_kwargs['marker']['color'] = 'steelblue'
    
    fig = go.Figure()
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
    
    # Set layout
    camera = camera_state if camera_state else {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': {'x': 1.5, 'y': 1.5, 'z': 1.5}
    }
    
    fig.update_layout(
        title=title,
        uirevision='camera',  # This ensures camera views stay in sync
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=camera,
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range)
        ),
        margin=dict(l=0, r=40, b=0, t=40),
        height=500,
    )
    
    return fig


def get_point_cloud_stats(
    points: torch.Tensor,
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> html.Ul:
    """Get statistical information about a point cloud.

    Args:
        points: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        List of html components with point cloud statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(points, torch.Tensor), f"points must be torch.Tensor, got {type(points)}"
    assert points.ndim == 2 and points.shape[1] >= 3, f"points must be (N, 3+), got {points.shape}"
    
    if change_map is not None:
        assert isinstance(change_map, torch.Tensor), f"change_map must be torch.Tensor, got {type(change_map)}"
        assert change_map.shape[0] == points.shape[0], f"change_map length {change_map.shape[0]} != points length {points.shape[0]}"
    
    # Convert to numpy for stats computation
    points_np = points.detach().cpu().numpy()
    stats_items = [
        html.Li(f"Total Points: {len(points_np)}"),
        html.Li(f"Dimensions: {points_np.shape[1]}"),
        html.Li(f"X Range: [{points_np[:, 0].min():.2f}, {points_np[:, 0].max():.2f}]"),
        html.Li(f"Y Range: [{points_np[:, 1].min():.2f}, {points_np[:, 1].max():.2f}]"),
        html.Li(f"Z Range: [{points_np[:, 2].min():.2f}, {points_np[:, 2].max():.2f}]"),
        html.Li(f"Center: [{points_np[:, 0].mean():.2f}, {points_np[:, 1].mean():.2f}, {points_np[:, 2].mean():.2f}]")
    ]

    # Add class distribution if change_map is provided
    if change_map is not None:
        unique_classes, class_counts = torch.unique(change_map, return_counts=True)
        unique_classes = unique_classes.cpu().numpy()
        class_counts = class_counts.cpu().numpy()
        total_points = change_map.numel()

        stats_items.append(html.Li("Class Distribution:"))
        class_list_items = []

        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / total_points) * 100
            cls_key = cls.item() if hasattr(cls, 'item') else cls
            class_name = class_names[cls_key] if class_names and cls_key in class_names else f"Class {cls_key}"
            class_list_items.append(
                html.Li(f"{class_name}: {count} points ({percentage:.2f}%)",
                       style={'marginLeft': '20px'})
            )

        stats_items.append(html.Ul(class_list_items))

    return html.Ul(stats_items)


def create_point_cloud_display(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "continuous",
    lod_config: Optional[Dict[str, Any]] = None,
    density_percentage: Optional[int] = None,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    **kwargs: Any
) -> go.Figure:
    """Create point cloud display with LOD optimization.
    
    Args:
        points: Point cloud positions tensor of shape [N, 3]
        colors: Optional color tensor of shape [N, 3] or [N, C]
        labels: Optional label tensor of shape [N]
        title: Title for the point cloud display
        point_size: Size of the points in visualization
        point_opacity: Opacity of the points in visualization
        camera_state: Optional camera position state for 3D view
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
    # CRITICAL: Input validation with fail-fast assertions
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"
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
    
    # Use create_point_cloud_figure implementation
    return create_point_cloud_figure(
        points=points,
        colors=colors,
        labels=labels,
        title=title,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
        lod_type=lod_type,
        lod_config=lod_config,
        density_percentage=density_percentage,
        point_cloud_id=point_cloud_id,
        axis_ranges=axis_ranges
    )


def get_point_cloud_display_stats(
    points: torch.Tensor,
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> html.Ul:
    """Get point cloud statistics for display.
    
    Args:
        points: Point cloud tensor of shape [N, 3+]
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names
        
    Returns:
        HTML list containing point cloud statistics
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    # Input validation
    assert isinstance(points, torch.Tensor), f"Expected torch.Tensor, got {type(points)}"
    assert points.ndim == 2, f"Expected 2D tensor [N,D], got shape {points.shape}"
    assert points.shape[1] >= 3, f"Expected at least 3 coordinates, got {points.shape[1]}"
    assert points.numel() > 0, f"Point cloud cannot be empty"
    
    if change_map is not None:
        assert isinstance(change_map, torch.Tensor), f"change_map must be torch.Tensor, got {type(change_map)}"
        assert change_map.shape[0] == points.shape[0], f"change_map length {change_map.shape[0]} != points length {points.shape[0]}"
    
    if class_names is not None:
        assert isinstance(class_names, dict), f"class_names must be dict, got {type(class_names)}"
    
    # Use get_point_cloud_stats implementation
    return get_point_cloud_stats(
        points=points,
        change_map=change_map,
        class_names=class_names
    )


def build_point_cloud_display_id(
    datapoint: Dict[str, Any], 
    component: str
) -> Tuple[str, int, str]:
    """Build structured point cloud ID for LOD caching.
    
    Args:
        datapoint: Contains meta_info with dataset context
        component: Point cloud component identifier
        
    Returns:
        Tuple of (dataset_name, datapoint_idx, component)
        
    Raises:
        AssertionError: If inputs don't meet requirements
    """
    assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
    assert isinstance(component, str), f"component must be str, got {type(component)}"
    
    # Use build_point_cloud_id implementation
    return build_point_cloud_id(datapoint=datapoint, component=component)