"""UI components for displaying point cloud registration dataset items."""
from typing import Tuple, Dict, Optional, Any
import random
import time
import numpy as np
import torch
from dash import dcc, html
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.point_cloud_ops import apply_transform, get_correspondences
from utils.point_cloud_ops.set_ops import pc_symmetric_difference
from utils.point_cloud_ops.set_ops.symmetric_difference import _normalize_points
from utils.point_cloud_ops.apply_transform import _normalize_transform
from data.viewer.utils.point_cloud import create_point_cloud_figure, get_point_cloud_stats


def create_union_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    lod_enabled: bool = True,
) -> go.Figure:
    """Create a visualization of the union of transformed source and target point clouds.

    Args:
        src_points: Transformed source point cloud [N, 3] or [1, N, 3]
        tgt_points: Target point cloud [M, 3] or [1, M, 3]
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        lod_enabled: Whether LOD optimization is enabled

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
        lod_enabled=lod_enabled,
        point_cloud_id="union_visualization",
    )


def create_symmetric_difference_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float = 0.05,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Create a visualization of the symmetric difference between transformed source and target point clouds.

    Args:
        src_points: Transformed source point cloud [N, 3] or [1, N, 3]
        tgt_points: Target point cloud [M, 3] or [1, M, 3]
        radius: Radius for computing symmetric difference
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state

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
        )
    else:
        # If no symmetric difference, show empty point cloud
        return create_point_cloud_figure(
            torch.zeros((1, 3), device=src_points_normalized.device),
            title="Symmetric Difference (Empty)",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state
        )


def create_correspondence_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    radius: float = 0.1,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Create a visualization of correspondences between transformed source and target point clouds.

    Args:
        src_points: Transformed source point cloud [N, 3] or [1, N, 3]
        tgt_points: Target point cloud [M, 3] or [1, M, 3]
        radius: Radius for finding correspondences
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state

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
        camera_state=camera_state
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


def display_pcr_datapoint_single(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    sym_diff_radius: float = 0.05,
    corr_radius: float = 0.1
) -> html.Div:
    """Display a single point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        sym_diff_radius: Radius for computing symmetric difference
        corr_radius: Radius for finding correspondences between point clouds

    Returns:
        html.Div containing the visualization
    """
    start_time = time.time()
    print(f"[PCR Display] Starting display_pcr_datapoint_single callback at {start_time:.4f}")
    # Check if the inputs have the expected structure
    inputs = datapoint['inputs']
    assert 'src_pc' in inputs and 'tgt_pc' in inputs, "Source point cloud (src_pc) and target point cloud (tgt_pc) must be present in the inputs"
    assert isinstance(inputs['src_pc'], dict) and isinstance(inputs['tgt_pc'], dict), "Point clouds must be dictionaries"
    assert 'pos' in inputs['src_pc'] and 'pos' in inputs['tgt_pc'], "Point clouds must have 'pos' field"

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

    # Create the point cloud views in parallel for better performance
    def create_source_figure():
        return create_point_cloud_figure(
            points=src_pc,
            colors=src_rgb,
            title="Source Point Cloud",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
        )

    def create_target_figure():
        return create_point_cloud_figure(
            points=tgt_pc,
            colors=tgt_rgb,
            title="Target Point Cloud",
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
        )

    def create_union_figure():
        return create_union_visualization(
            src_pc_transformed,
            tgt_pc,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
        )

    def create_sym_diff_figure():
        return create_symmetric_difference_visualization(
            src_pc_transformed,
            tgt_pc,
            radius=sym_diff_radius,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
        )

    # Create figures in parallel
    figure_tasks = [
        create_source_figure,
        create_target_figure,
        create_union_figure,
        create_sym_diff_figure,
    ]

    figures = [None] * len(figure_tasks)  # Pre-allocate list to maintain order
    
    figure_start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(task_func): idx 
            for idx, task_func in enumerate(figure_tasks)
        }
        
        # Collect results in order
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            figures[idx] = future.result()
    figure_time = time.time() - figure_start
    print(f"[PCR Display] Figure creation took {figure_time:.4f}s")

    # TODO: Add correspondence visualization
    # # 5. Correspondence visualization
    # figures.append(create_correspondence_visualization(
    #     src_pc_transformed,
    #     tgt_pc,
    #     radius=corr_radius,
    #     point_size=point_size,
    #     point_opacity=point_opacity,
    #     camera_state=camera_state,
    # ))

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

    # Get point cloud statistics
    src_stats_children = get_point_cloud_stats(src_pc)
    tgt_stats_children = get_point_cloud_stats(tgt_pc)

    # Format the transformation matrix as a string
    transform_str = "Transform Matrix:\n"
    for i in range(4):
        row = [f"{transform_normalized[i, j]:.4f}" for j in range(4)]
        transform_str += "  ".join(row) + "\n"

    # Create a grid layout for the five figures
    result = html.Div([
        html.H3("Point Cloud Registration Visualization"),
        html.Div([
            # Create grid items using a for loop
            *[html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': i},
                    figure=fig,
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'}) for i, fig in enumerate(figures)]
        ], style={'display': 'flex', 'flex-wrap': 'wrap'}),

        # Display transform information
        html.Div([
            html.H4("Transform Information:"),
            html.Pre(transform_str),
            html.P(f"Rotation Angle: {rotation_angle:.2f} degrees"),
            html.P(f"Translation Magnitude: {translation_magnitude:.4f}")
        ], style={'margin-top': '20px'}),

        # Point cloud statistics
        html.Div([
            html.Div([
                html.H4("Source Point Cloud Statistics:"),
                html.Div(src_stats_children)
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '2%'}),
            
            html.Div([
                html.H4("Target Point Cloud Statistics:"),
                html.Div(tgt_stats_children)
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
        ], style={'margin-top': '20px'})
    ])
    
    total_time = time.time() - start_time
    print(f"[PCR Display] Total display_pcr_datapoint_single time: {total_time:.4f}s")
    
    return result


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


def display_pcr_datapoint_batched(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    sym_diff_radius: float = 0.05,
    corr_radius: float = 0.1
) -> html.Div:
    """Display a batched point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        sym_diff_radius: Radius for computing symmetric difference
        corr_radius: Radius for finding correspondences between point clouds

    Returns:
        html.Div containing the visualization
    """
    start_time = time.time()
    print(f"[PCR Display] Starting display_pcr_datapoint_batched callback at {start_time:.4f}")
    inputs = datapoint['inputs']
    figures = []

    # Process each level in the hierarchy
    for level in range(len(inputs['points'])):
        # Split points into source and target
        src_points, tgt_points = split_points_by_lengths(
            inputs['points'][level], inputs.get('lengths', inputs['stack_lengths'])[level],
        )

        # For top level (level 0), show all visualizations
        if level == 0:
            # Create visualization functions
            def create_source_figure():
                return create_point_cloud_figure(
                    points=src_points,
                    title=f"Source Point Cloud (Level {level})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                )

            def create_target_figure():
                return create_point_cloud_figure(
                    points=tgt_points,
                    title=f"Target Point Cloud (Level {level})",
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                )

            def create_union_figure():
                union_fig = create_union_visualization(
                    src_points,
                    tgt_points,
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                )
                union_fig.update_layout(title=f"Union (Level {level})")
                return union_fig

            def create_sym_diff_figure():
                sym_diff_fig = create_symmetric_difference_visualization(
                    src_points,
                    tgt_points,
                    radius=sym_diff_radius,
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                )
                sym_diff_fig.update_layout(title=f"Symmetric Difference (Level {level})")
                return sym_diff_fig

            # Create figures in parallel
            figure_tasks = [
                create_source_figure,
                create_target_figure,
                create_union_figure,
                create_sym_diff_figure,
            ]

            level_figures = [None] * len(figure_tasks)
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(task_func): idx 
                    for idx, task_func in enumerate(figure_tasks)
                }
                
                # Collect results in order
                for future in as_completed(future_to_index):
                    idx = future_to_index[future]
                    level_figures[idx] = future.result()
            
            figures.extend(level_figures)

            # TODO: Add correspondence visualization
            # # Correspondence visualization
            # corr_fig = create_correspondence_visualization(
            #     src_points,
            #     tgt_points,
            #     radius=corr_radius,
            #     point_size=point_size,
            #     point_opacity=point_opacity,
            #     camera_state=camera_state,
            # )
            # corr_fig.update_layout(title=f"Point Cloud Correspondences (Level {level})")
            # figures.append(corr_fig)
        else:
            # For lower levels, only show source and target
            figures.append(create_point_cloud_figure(
                points=src_points,
                title=f"Source Point Cloud (Level {level})",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state
            ))

            figures.append(create_point_cloud_figure(
                points=tgt_points,
                title=f"Target Point Cloud (Level {level})",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state
            ))

    # Create grid layout
    grid_items = []
    for i, fig in enumerate(figures):
        grid_items.append(
            html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': i},
                    figure=fig,
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'})
        )

    result = html.Div([
        html.H3("Point Cloud Registration Visualization (Hierarchical)"),
        html.Div(grid_items, style={'display': 'flex', 'flex-wrap': 'wrap'})
    ])
    
    total_time = time.time() - start_time
    print(f"[PCR Display] Total display_pcr_datapoint_batched time: {total_time:.4f}s")
    
    return result


def display_pcr_datapoint(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    sym_diff_radius: float = 0.05,
    corr_radius: float = 0.1
) -> html.Div:
    """Display a point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        sym_diff_radius: Radius for computing symmetric difference
        corr_radius: Radius for finding correspondences between point clouds

    Returns:
        html.Div containing the visualization
    """
    start_time = time.time()
    print(f"[PCR Display] Starting display_pcr_datapoint callback at {start_time:.4f}")
    inputs = datapoint['inputs']

    # Check if we have hierarchical data (from collators)
    if 'points' in inputs and ('lengths' in inputs or 'stack_lengths' in inputs):
        result = display_pcr_datapoint_batched(
            datapoint,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            sym_diff_radius=sym_diff_radius,
            corr_radius=corr_radius,
        )
    else:
        result = display_pcr_datapoint_single(
            datapoint,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            sym_diff_radius=sym_diff_radius,
            corr_radius=corr_radius,
        )
    
    total_time = time.time() - start_time
    print(f"[PCR Display] Total display_pcr_datapoint time: {total_time:.4f}s")
    
    return result
