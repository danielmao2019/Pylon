"""UI components for displaying point cloud registration dataset items."""
from typing import Tuple, Dict, Optional, Any, Union
import random
import numpy as np
import torch
from dash import html
import plotly.graph_objects as go
from utils.point_cloud_ops import apply_transform, get_correspondences
from utils.point_cloud_ops.set_ops import pc_symmetric_difference
from utils.point_cloud_ops.set_ops.symmetric_difference import _normalize_points
from utils.point_cloud_ops.apply_transform import _normalize_transform
from data.viewer.utils.atomic_displays.point_cloud_display import create_point_cloud_figure, get_point_cloud_stats, build_point_cloud_id
from data.viewer.utils.display_utils import DisplayStyles, ParallelFigureCreator, create_figure_grid
from data.viewer.utils.structure_validation import validate_pcr_structure


def create_union_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "continuous",
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    density_percentage: int = 100,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> go.Figure:
    """Create a visualization of the union of transformed source and target point clouds.

    Args:
        src_points: Transformed source point cloud [N, 3] or [1, N, 3]
        tgt_points: Target point cloud [M, 3] or [1, M, 3]
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        point_cloud_id: Unique identifier for LOD caching
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)

    Returns:
        Plotly figure showing the union visualization
    """
    # Normalize points to unbatched format
    src_points_normalized = _normalize_points(src_points)
    tgt_points_normalized = _normalize_points(tgt_points)
    
    # Combine points
    union_points = torch.cat([src_points_normalized, tgt_points_normalized], dim=0)

    # Create colors for union (red for source, blue for target)
    src_colors = torch.zeros((len(src_points_normalized), 3), device=src_points_normalized.device)
    src_colors[:, 0] = 1.0  # Red for source
    tgt_colors = torch.zeros((len(tgt_points_normalized), 3), device=tgt_points_normalized.device)
    tgt_colors[:, 2] = 1.0  # Blue for target
    union_colors = torch.cat([src_colors, tgt_colors], dim=0)

    return create_point_cloud_figure(
        points=union_points,
        colors=union_colors,
        title="Union (Transformed Source + Target)",
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
        lod_type=lod_type,
        density_percentage=density_percentage,
        point_cloud_id=point_cloud_id,
        axis_ranges=axis_ranges,
    )


def create_symmetric_difference_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float = 0.05,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "continuous",
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
    density_percentage: int = 100,
    axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> go.Figure:
    """Create a visualization of the symmetric difference between transformed source and target point clouds.

    Args:
        src_points: Transformed source point cloud [N, 3] or [1, N, 3]
        tgt_points: Target point cloud [M, 3] or [1, M, 3]
        radius: Radius for computing symmetric difference
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        point_cloud_id: Unique identifier for LOD caching
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)

    Returns:
        Plotly figure showing the symmetric difference visualization
    """
    # Normalize points to unbatched format
    src_points_normalized = _normalize_points(src_points)
    tgt_points_normalized = _normalize_points(tgt_points)
    
    # Find points in symmetric difference
    src_indices, tgt_indices = pc_symmetric_difference(src_points_normalized, tgt_points_normalized, radius)

    if len(src_indices) > 0 or len(tgt_indices) > 0:
        # Extract points in symmetric difference
        src_diff = src_points_normalized[src_indices]
        tgt_diff = tgt_points_normalized[tgt_indices]

        # Combine the points
        sym_diff_points = torch.cat([src_diff, tgt_diff], dim=0)

        # Create colors for symmetric difference (red for source, blue for target)
        src_colors = torch.zeros((len(src_indices), 3), device=src_diff.device)
        src_colors[:, 0] = 1.0  # Red for source
        tgt_colors = torch.zeros((len(tgt_indices), 3), device=tgt_diff.device)
        tgt_colors[:, 2] = 1.0  # Blue for target
        sym_diff_colors = torch.cat([src_colors, tgt_colors], dim=0)

        return create_point_cloud_figure(
            points=sym_diff_points,
            colors=sym_diff_colors,
            title="Symmetric Difference",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            density_percentage=density_percentage,
            point_cloud_id=point_cloud_id,
            axis_ranges=axis_ranges,
        )
    else:
        # If no symmetric difference, show empty point cloud
        return create_point_cloud_figure(
            torch.zeros((1, 3), device=src_points_normalized.device),
            title="Symmetric Difference (Empty)",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            density_percentage=density_percentage,
            point_cloud_id=point_cloud_id,
            axis_ranges=axis_ranges,
        )


def create_correspondence_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float = 0.1,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_type: str = "continuous",
    density_percentage: int = 100,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
) -> go.Figure:
    """Create a visualization of correspondences between transformed source and target point clouds.

    Args:
        src_points: Transformed source point cloud [N, 3] or [1, N, 3]
        tgt_points: Target point cloud [M, 3] or [1, M, 3]
        radius: Radius for finding correspondences
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)
        point_cloud_id: Unique identifier for LOD caching

    Returns:
        Plotly figure showing the correspondence visualization
    """
    # Normalize points to unbatched format
    src_points_normalized = _normalize_points(src_points)
    tgt_points_normalized = _normalize_points(tgt_points)
    
    src_points_np = src_points_normalized.cpu().numpy()
    tgt_points_np = tgt_points_normalized.cpu().numpy()

    # Find correspondences based on radius
    correspondences = get_correspondences(src_points_normalized, tgt_points_normalized, None, radius)

    # Create figure with both point clouds
    corr_fig = create_point_cloud_figure(
        points=src_points_normalized,
        title="Point Cloud Correspondences",
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
        lod_type=lod_type,
        density_percentage=density_percentage,
        point_cloud_id=point_cloud_id,
    )

    # Add target points
    corr_fig.add_trace(go.Scatter3d(
        x=tgt_points_np[:, 0],
        y=tgt_points_np[:, 1],
        z=tgt_points_np[:, 2],
        mode='markers',
        marker=dict(size=point_size, color='red', opacity=point_opacity),
        name='Target Points'
    ))

    # Create list of correspondence line traces
    correspondence_traces = []
    for src_idx, tgt_idx in correspondences:
        src_point = src_points_np[src_idx]
        tgt_point = tgt_points_np[tgt_idx]
        correspondence_traces.append(go.Scatter3d(
            x=[src_point[0], tgt_point[0]],
            y=[src_point[1], tgt_point[1]],
            z=[src_point[2], tgt_point[2]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

    if len(correspondence_traces) > 10:
        correspondence_traces = random.sample(correspondence_traces, 10)

    # Add all correspondence traces at once
    if correspondence_traces:
        corr_fig.add_traces(correspondence_traces)

    return corr_fig


def _compute_transform_info(transform: torch.Tensor) -> Dict[str, Any]:
    """Compute transform information including rotation angle and translation magnitude."""
    # Normalize transform to handle batched case
    transform_normalized = _normalize_transform(transform, torch.Tensor)
    
    # Compute rotation angle and translation magnitude
    rotation_matrix = transform_normalized[:3, :3]
    translation_vector = transform_normalized[:3, 3]

    # Compute rotation angle using the trace of the rotation matrix
    trace = torch.trace(rotation_matrix)
    rotation_angle = torch.acos((trace - 1) / 2) * 180 / np.pi  # Convert to degrees

    # Compute translation magnitude
    translation_magnitude = torch.norm(translation_vector)

    # Format the transformation matrix as a string
    transform_str = "Transform Matrix:\n"
    for i in range(4):
        row = [f"{transform_normalized[i, j]:.4f}" for j in range(4)]
        transform_str += "  ".join(row) + "\n"

    return {
        'transform_str': transform_str,
        'rotation_angle': rotation_angle,
        'translation_magnitude': translation_magnitude
    }


def _create_transform_info_section(transform_info: Dict[str, Any]) -> html.Div:
    """Create transform information section."""
    return html.Div([
        html.H4("Transform Information:"),
        html.Pre(transform_info['transform_str']),
        html.P(f"Rotation Angle: {transform_info['rotation_angle']:.2f} degrees"),
        html.P(f"Translation Magnitude: {transform_info['translation_magnitude']:.4f}")
    ], style={'margin-top': '20px'})


def _create_statistics_section(src_stats_children: Any, tgt_stats_children: Any) -> html.Div:
    """Create point cloud statistics section."""
    return html.Div([
        html.Div([
            html.H4("Source Point Cloud Statistics:"),
            html.Div(src_stats_children)
        ], style=DisplayStyles.GRID_ITEM_48_MARGIN),
        
        html.Div([
            html.H4("Target Point Cloud Statistics:"),
            html.Div(tgt_stats_children)
        ], style=DisplayStyles.GRID_ITEM_48_NO_MARGIN)
    ], style={'margin-top': '20px'})


def _format_value(key: str, value: Any) -> str:
    """Format a value for display based on key name and value type."""
    if isinstance(value, list) and len(value) == 3 and all(isinstance(x, (int, float)) for x in value):
        # Handle 3D vectors with context-specific formatting
        if 'angle' in key.lower():
            return f"[{value[0]:.2f}°, {value[1]:.2f}°, {value[2]:.2f}°]"
        else:
            return f"[{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}]"
    elif isinstance(value, float):
        return f"{value:.4f}"
    else:
        return str(value)


def _dict_to_html_list(data: Dict[str, Any], key_name: str = None) -> html.Div:
    """Convert a dictionary to HTML list structure."""
    items = []
    
    if key_name:
        items.append(html.H5(f"{key_name}:", style={'margin-top': '15px', 'margin-bottom': '5px'}))
    
    list_items = []
    for key, value in data.items():
        if isinstance(value, dict):
            # Nested dictionary - create sub-list
            items.append(_dict_to_html_list(value, key))
        else:
            # Simple key-value pair
            formatted_value = _format_value(key, value)
            
            # Special styling for overlap (important PCR metric)
            if key == 'overlap':
                list_items.append(html.Li(f"{key}: {formatted_value}", 
                                        style={'font-weight': 'bold', 'color': '#2E86AB'}))
            else:
                list_items.append(html.Li(f"{key}: {formatted_value}"))
    
    if list_items:
        items.append(html.Ul(list_items, style={'margin-left': '20px', 'margin-top': '5px'}))
    
    return html.Div(items)


def _create_meta_info_section(meta_info: Dict[str, Any]) -> html.Div:
    """Create meta information section displaying dataset metadata."""
    if not meta_info:
        return html.Div([
            html.H4("Datapoint Meta Information:"),
            html.P("No meta information available")
        ], style={'margin-top': '20px'})
    
    # Convert the entire meta_info dict to HTML lists
    return html.Div([
        html.H4("Datapoint Meta Information:"),
        _dict_to_html_list(meta_info)
    ], style={'margin-top': '20px'})


def display_pcr_datapoint_single(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    sym_diff_radius: float = 0.05,
    corr_radius: float = 0.1,
    lod_type: str = "continuous",
    density_percentage: int = 100
) -> html.Div:
    """Display a single point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        sym_diff_radius: Radius for computing symmetric difference
        corr_radius: Radius for finding correspondences between point clouds
        lod_type: Type of LOD ("continuous", "discrete", or "none")

    Returns:
        html.Div containing the visualization
    """
    # Validate structure and inputs (includes all basic validation)
    validate_pcr_structure(datapoint)
    
    inputs = datapoint['inputs']

    # Extract point clouds
    src_pc = inputs['src_pc']['pos']  # Source point cloud
    tgt_pc = inputs['tgt_pc']['pos']  # Target point cloud

    # Extract RGB colors if available
    src_rgb = inputs['src_pc'].get('rgb')
    tgt_rgb = inputs['tgt_pc'].get('rgb')

    # Extract transform if available
    transform = datapoint['labels'].get('transform')
    if transform is None:
        transform = torch.eye(4)  # Default to identity transform if not provided

    # Apply transform to source point cloud
    src_pc_transformed = apply_transform(src_pc, transform)

    # Compute unified axis ranges across all point clouds for consistent scaling
    all_points = [src_pc, tgt_pc, src_pc_transformed]
    x_coords = torch.cat([pc[:, 0] for pc in all_points])
    y_coords = torch.cat([pc[:, 1] for pc in all_points])
    z_coords = torch.cat([pc[:, 2] for pc in all_points])
    
    # Add small padding for better visualization
    padding = 0.05  # 5% padding
    x_range_unified = [x_coords.min().item(), x_coords.max().item()]
    y_range_unified = [y_coords.min().item(), y_coords.max().item()]
    z_range_unified = [z_coords.min().item(), z_coords.max().item()]
    
    # Apply padding
    x_pad = (x_range_unified[1] - x_range_unified[0]) * padding
    y_pad = (y_range_unified[1] - y_range_unified[0]) * padding
    z_pad = (z_range_unified[1] - z_range_unified[0]) * padding
    
    unified_axis_ranges = {
        'x': (x_range_unified[0] - x_pad, x_range_unified[1] + x_pad),
        'y': (y_range_unified[0] - y_pad, y_range_unified[1] + y_pad),
        'z': (z_range_unified[0] - z_pad, z_range_unified[1] + z_pad)
    }

    # Define figure creation tasks
    figure_tasks = [
        lambda: create_point_cloud_figure(
            points=src_pc,
            colors=src_rgb,
            title="Source Point Cloud",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            density_percentage=density_percentage,
            point_cloud_id=build_point_cloud_id(datapoint, "source"),
            axis_ranges=unified_axis_ranges,
        ),
        lambda: create_point_cloud_figure(
            points=tgt_pc,
            colors=tgt_rgb,
            title="Target Point Cloud",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            density_percentage=density_percentage,
            point_cloud_id=build_point_cloud_id(datapoint, "target"),
            axis_ranges=unified_axis_ranges,
        ),
        lambda: create_union_visualization(
            src_pc_transformed,
            tgt_pc,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            point_cloud_id=build_point_cloud_id(datapoint, "union"),
            density_percentage=density_percentage,
            axis_ranges=unified_axis_ranges,
        ),
        lambda: create_symmetric_difference_visualization(
            src_pc_transformed,
            tgt_pc,
            radius=sym_diff_radius,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            point_cloud_id=build_point_cloud_id(datapoint, "sym_diff"),
            density_percentage=density_percentage,
            axis_ranges=unified_axis_ranges,
        ),
    ]

    # Create figures in parallel using centralized utility
    figure_creator = ParallelFigureCreator(max_workers=4, enable_timing=False)
    figures = figure_creator.create_figures_parallel(figure_tasks)

    # TODO: Add correspondence visualization
    # figures.append(create_correspondence_visualization(
    #     src_pc_transformed,
    #     tgt_pc,
    #     radius=corr_radius,
    #     point_size=point_size,
    #     point_opacity=point_opacity,
    #     camera_state=camera_state,
    # ))

    # Compute transform information
    transform_info = _compute_transform_info(transform)
    
    # Get point cloud statistics
    src_stats_children = get_point_cloud_stats(src_pc)
    tgt_stats_children = get_point_cloud_stats(tgt_pc)

    # Create layout using centralized utilities
    grid_items = create_figure_grid(figures, width_style="50%", height_style="520px")
    
    return html.Div([
        html.H3("Point Cloud Registration Visualization"),
        html.Div(grid_items, style=DisplayStyles.FLEX_WRAP),
        _create_transform_info_section(transform_info),
        _create_statistics_section(src_stats_children, tgt_stats_children),
        _create_meta_info_section(datapoint.get('meta_info', {}))
    ])


def split_points_by_lengths(points: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split concatenated points into source and target using lengths.

    Args:
        points: Concatenated points tensor [src_points, tgt_points]
        lengths: Lengths tensor indicating split point

    Returns:
        Tuple of (source_points, target_points)
    """
    total_length = lengths[0]
    src_points = points[:total_length//2]
    tgt_points = points[total_length//2:total_length]
    return src_points, tgt_points


def _create_union_with_title(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    title: str,
    point_size: float,
    point_opacity: float,
    camera_state: Optional[Dict[str, Any]],
    lod_type: str,
    density_percentage: int = 100,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
) -> go.Figure:
    """Create union visualization with custom title."""
    union_fig = create_union_visualization(
        src_points=src_points,
        tgt_points=tgt_points,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
        lod_type=lod_type,
        point_cloud_id=point_cloud_id,
        density_percentage=density_percentage
    )
    union_fig.update_layout(title=title)
    return union_fig


def _create_sym_diff_with_title(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    title: str,
    radius: float,
    point_size: float,
    point_opacity: float,
    camera_state: Optional[Dict[str, Any]],
    lod_type: str,
    density_percentage: int = 100,
    point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
) -> go.Figure:
    """Create symmetric difference visualization with custom title."""
    sym_diff_fig = create_symmetric_difference_visualization(
        src_points=src_points,
        tgt_points=tgt_points,
        radius=radius,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
        lod_type=lod_type,
        point_cloud_id=point_cloud_id,
        density_percentage=density_percentage
    )
    sym_diff_fig.update_layout(title=title)
    return sym_diff_fig


def display_pcr_datapoint_batched(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    sym_diff_radius: float = 0.05,
    corr_radius: float = 0.1,
    lod_type: str = "continuous",
    density_percentage: int = 100
) -> html.Div:
    """Display a batched point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        sym_diff_radius: Radius for computing symmetric difference
        corr_radius: Radius for finding correspondences between point clouds
        lod_type: Type of LOD ("continuous", "discrete", or "none")

    Returns:
        html.Div containing the visualization
    """
    # Validate structure and inputs (includes all basic validation)
    validate_pcr_structure(datapoint)
    
    inputs = datapoint['inputs']
    all_figures = []

    # Process each level in the hierarchy
    for level in range(len(inputs['points'])):
        # Split points into source and target
        src_points, tgt_points = split_points_by_lengths(
            inputs['points'][level], inputs.get('lengths', inputs['stack_lengths'])[level],
        )

        # For top level (level 0), show all visualizations
        if level == 0:
            figure_tasks = [
                lambda src=src_points, lvl=level: create_point_cloud_figure(
                    points=src,
                    title=f"Source Point Cloud (Level {lvl})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, f"source_batch_{lvl}"),
                ),
                lambda tgt=tgt_points, lvl=level: create_point_cloud_figure(
                    points=tgt,
                    title=f"Target Point Cloud (Level {lvl})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, f"target_batch_{lvl}"),
                ),
                lambda src=src_points, tgt=tgt_points, lvl=level: _create_union_with_title(
                    src_points=src,
                    tgt_points=tgt,
                    title=f"Union (Level {lvl})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, f"union_batch_{lvl}")
                ),
                lambda src=src_points, tgt=tgt_points, lvl=level: _create_sym_diff_with_title(
                    src_points=src,
                    tgt_points=tgt,
                    title=f"Symmetric Difference (Level {lvl})",
                    radius=sym_diff_radius,
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, f"sym_diff_batch_{lvl}")
                ),
            ]

            # Create figures in parallel using centralized utility
            figure_creator = ParallelFigureCreator(max_workers=4, enable_timing=False)
            level_figures = figure_creator.create_figures_parallel(figure_tasks)
            all_figures.extend(level_figures)

            # TODO: Add correspondence visualization
            # corr_fig = create_correspondence_visualization(
            #     src_points, tgt_points, radius=corr_radius, point_size=point_size,
            #     point_opacity=point_opacity, camera_state=camera_state,
            # )
            # corr_fig.update_layout(title=f"Point Cloud Correspondences (Level {level})")
            # all_figures.append(corr_fig)
        else:
            # For lower levels, only show source and target
            all_figures.extend([
                create_point_cloud_figure(
                    points=src_points,
                    title=f"Source Point Cloud (Level {level})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, f"source_batch_{level}"),
                ),
                create_point_cloud_figure(
                    points=tgt_points,
                    title=f"Target Point Cloud (Level {level})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, f"target_batch_{level}"),
                )
            ])

    # Create grid layout using centralized utilities
    grid_items = create_figure_grid(all_figures, width_style="50%", height_style="520px")
    
    return html.Div([
        html.H3("Point Cloud Registration Visualization (Hierarchical)"),
        html.Div(grid_items, style=DisplayStyles.FLEX_WRAP),
        _create_meta_info_section(datapoint.get('meta_info', {}))
    ])


def display_pcr_datapoint(
    datapoint: Dict[str, Any],
    camera_state: Optional[Dict[str, Any]] = None,
    point_size: float = 2,
    point_opacity: float = 0.8,
    sym_diff_radius: float = 0.05,
    corr_radius: float = 0.1,
    lod_type: str = "continuous",
    density_percentage: int = 100
) -> html.Div:
    """Display a point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        sym_diff_radius: Radius for computing symmetric difference
        corr_radius: Radius for finding correspondences between point clouds
        lod_type: Type of LOD ("continuous", "discrete", or "none")
        density_percentage: Percentage of points to display when lod_type is "none" (1-100)

    Returns:
        html.Div containing the visualization
    """
    # Validate structure and inputs (includes all basic validation)
    validate_pcr_structure(datapoint)
    
    inputs = datapoint['inputs']

    # Check if we have hierarchical data (from collators)
    if 'points' in inputs and ('lengths' in inputs or 'stack_lengths' in inputs):
        return display_pcr_datapoint_batched(
            datapoint,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            sym_diff_radius=sym_diff_radius,
            corr_radius=corr_radius,
            lod_type=lod_type,
            density_percentage=density_percentage,
        )
    else:
        return display_pcr_datapoint_single(
            datapoint,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            sym_diff_radius=sym_diff_radius,
            corr_radius=corr_radius,
            lod_type=lod_type,
            density_percentage=density_percentage,
        )
