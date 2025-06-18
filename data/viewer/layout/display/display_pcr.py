"""UI components for displaying point cloud registration dataset items."""
from typing import Tuple, Dict, Optional, Any
import numpy as np
import torch
from dash import dcc, html
import plotly.graph_objects as go
from utils.point_cloud_ops import apply_transform
from utils.point_cloud_ops.set_ops import pc_symmetric_difference
from data.viewer.utils.point_cloud import create_point_cloud_figure


def create_union_visualization(
    src_points: torch.Tensor,
    tgt_points: torch.Tensor,
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
) -> go.Figure:
    """Create a visualization of the union of transformed source and target point clouds.

    Args:
        src_pc_transformed: Transformed source point cloud
        tgt_pc: Target point cloud
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state

    Returns:
        Plotly figure showing the union visualization
    """
    # Combine points
    union_points = torch.cat([src_points, tgt_points], dim=0)

    # Create colors for union (red for source, blue for target)
    src_colors = torch.zeros((len(src_points), 3), device=src_points.device)
    src_colors[:, 0] = 1.0  # Red for source
    tgt_colors = torch.zeros((len(tgt_points), 3), device=tgt_points.device)
    tgt_colors[:, 2] = 1.0  # Blue for target
    union_colors = torch.cat([src_colors, tgt_colors], dim=0)

    return create_point_cloud_figure(
        points=union_points,
        colors=union_colors,
        title="Union (Transformed Source + Target)",
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
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
        src_pc_transformed: Transformed source point cloud
        tgt_pc: Target point cloud
        radius: Radius for computing symmetric difference
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state

    Returns:
        Plotly figure showing the symmetric difference visualization
    """
    # Find points in symmetric difference
    src_indices, tgt_indices = pc_symmetric_difference(src_points, tgt_points, radius)

    if len(src_indices) > 0 or len(tgt_indices) > 0:
        # Extract points in symmetric difference
        src_diff = src_points[src_indices]
        tgt_diff = tgt_points[tgt_indices]

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
            torch.zeros((1, 3), device=src_points.device),
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
        src_pc_transformed: Transformed source point cloud
        tgt_pc: Target point cloud
        correspondence_radius: Radius for finding correspondences
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state

    Returns:
        Plotly figure showing the correspondence visualization
    """
    # Convert to numpy if needed
    src_points_np = src_points.cpu().numpy()
    tgt_points_np = tgt_points.cpu().numpy()

    # Find correspondences based on radius
    correspondences = []
    for i, src_point in enumerate(src_points_np):
        distances = np.linalg.norm(tgt_points_np - src_point, axis=1)
        matches = np.where(distances < radius)[0]
        for match in matches:
            correspondences.append((i, match))

    # Create figure with both point clouds
    corr_fig = create_point_cloud_figure(
        points=src_points,
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

    # Add correspondence lines
    for src_idx, tgt_idx in correspondences:
        src_point = src_points_np[src_idx]
        tgt_point = tgt_points_np[tgt_idx]

        corr_fig.add_trace(go.Scatter3d(
            x=[src_point[0], tgt_point[0]],
            y=[src_point[1], tgt_point[1]],
            z=[src_point[2], tgt_point[2]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False
        ))

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

    # Create the five point cloud views
    figures = []

    # 1. Source point cloud (original)
    figures.append(create_point_cloud_figure(
        points=src_pc,
        colors=src_rgb,
        title="Source Point Cloud",
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
    ))

    # 2. Target point cloud
    figures.append(create_point_cloud_figure(
        points=tgt_pc,
        colors=tgt_rgb,
        title="Target Point Cloud",
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
    ))

    # 3. Union visualization
    figures.append(create_union_visualization(
        src_pc_transformed,
        tgt_pc,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
    ))

    # 4. Symmetric difference visualization
    figures.append(create_symmetric_difference_visualization(
        src_pc_transformed,
        tgt_pc,
        radius=sym_diff_radius,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
    ))

    # 5. Correspondence visualization
    figures.append(create_correspondence_visualization(
        src_pc_transformed,
        tgt_pc,
        radius=corr_radius,
        point_size=point_size,
        point_opacity=point_opacity,
        camera_state=camera_state,
    ))

    # Compute rotation angle and translation magnitude
    rotation_matrix = transform[:3, :3]
    translation_vector = transform[:3, 3]

    # Compute rotation angle using the trace of the rotation matrix
    trace = torch.trace(rotation_matrix)
    rotation_angle = torch.acos((trace - 1) / 2) * 180 / np.pi  # Convert to degrees

    # Compute translation magnitude
    translation_magnitude = torch.norm(translation_vector)

    # Format the transformation matrix as a string
    transform_str = "Transform Matrix:\n"
    for i in range(4):
        row = [f"{transform[i, j]:.4f}" for j in range(4)]
        transform_str += "  ".join(row) + "\n"

    # Create a grid layout for the five figures
    return html.Div([
        html.H3("Point Cloud Registration Visualization"),
        html.Div([
            html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': 0},
                    figure=figures[0],
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': 1},
                    figure=figures[1],
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': 2},
                    figure=figures[2],
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': 3},
                    figure=figures[3],
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id={'type': 'point-cloud-graph', 'index': 4},
                    figure=figures[4],
                    style={'height': '400px'}
                )
            ], style={'width': '50%', 'display': 'inline-block'})
        ], style={'display': 'flex', 'flex-wrap': 'wrap'}),

        # Display transform information
        html.Div([
            html.H4("Transform Information:"),
            html.Pre(transform_str),
            html.P(f"Rotation Angle: {rotation_angle:.2f} degrees"),
            html.P(f"Translation Magnitude: {translation_magnitude:.4f}")
        ], style={'margin-top': '20px'})
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
            # Source point cloud
            figures.append(create_point_cloud_figure(
                points=src_points,
                title=f"Source Point Cloud (Level {level})",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
            ))

            # Target point cloud
            figures.append(create_point_cloud_figure(
                points=tgt_points,
                title=f"Target Point Cloud (Level {level})",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
            ))

            # Union visualization
            union_fig = create_union_visualization(
                src_points,
                tgt_points,
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
            )
            union_fig.update_layout(title=f"Union (Level {level})")
            figures.append(union_fig)

            # Symmetric difference visualization
            sym_diff_fig = create_symmetric_difference_visualization(
                src_points,
                tgt_points,
                radius=sym_diff_radius,
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
            )
            sym_diff_fig.update_layout(title=f"Symmetric Difference (Level {level})")
            figures.append(sym_diff_fig)

            # Correspondence visualization
            corr_fig = create_correspondence_visualization(
                src_points,
                tgt_points,
                radius=corr_radius,
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
            )
            corr_fig.update_layout(title=f"Point Cloud Correspondences (Level {level})")
            figures.append(corr_fig)
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

    return html.Div([
        html.H3("Point Cloud Registration Visualization (Hierarchical)"),
        html.Div(grid_items, style={'display': 'flex', 'flex-wrap': 'wrap'})
    ])


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
        )
    else:
        return display_pcr_datapoint_single(
            datapoint,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            sym_diff_radius=sym_diff_radius,
            corr_radius=corr_radius,
        )
