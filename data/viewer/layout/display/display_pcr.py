"""UI components for displaying point cloud registration dataset items."""
from typing import Tuple, Dict, Optional, Union, Any
import numpy as np
import torch
from dash import dcc, html
from utils.point_cloud_ops import apply_transform
from utils.point_cloud_ops.set_ops import pc_symmetric_difference
from data.viewer.utils.point_cloud import tensor_to_point_cloud, create_3d_figure, get_3d_stats


def display_pcr_datapoint_single(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    radius: float = 0.05
) -> html.Div:
    """Display a single point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        radius: Radius for computing symmetric difference

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

    # Compute symmetric difference
    src_indices, tgt_indices = pc_symmetric_difference(src_pc_transformed, tgt_pc, radius)

    # Create the four point cloud views
    figures = []

    # 1. Source point cloud (original)
    figures.append(create_3d_figure(
        src_pc,
        colors=src_rgb,
        title="Source Point Cloud",
        point_size=point_size,
        opacity=point_opacity,
        camera_state=camera_state,
        colorscale=None if src_rgb is not None else 'Viridis'
    ))

    # 2. Target point cloud
    figures.append(create_3d_figure(
        tgt_pc,
        colors=tgt_rgb,
        title="Target Point Cloud",
        point_size=point_size,
        opacity=point_opacity,
        camera_state=camera_state,
        colorscale=None if tgt_rgb is not None else 'Viridis'
    ))

    # 3. Union of transformed source and target
    union_pc = torch.cat([src_pc_transformed, tgt_pc], dim=0)

    # Create colors for union (red for source, blue for target)
    src_colors = torch.ones((len(src_pc_transformed), 3), device=src_pc_transformed.device)
    src_colors[:, 0] = 1.0  # Red for source
    src_colors[:, 1] = 0.0
    src_colors[:, 2] = 0.0

    tgt_colors = torch.ones((len(tgt_pc), 3), device=tgt_pc.device)
    tgt_colors[:, 0] = 0.0  # Blue for target
    tgt_colors[:, 1] = 0.0
    tgt_colors[:, 2] = 1.0

    union_colors = torch.cat([src_colors, tgt_colors], dim=0)

    figures.append(create_3d_figure(
        union_pc,
        colors=union_colors,
        title="Union (Transformed Source + Target)",
        point_size=point_size,
        opacity=point_opacity,
        camera_state=camera_state,
        colorscale=None  # Use custom colors
    ))

    # 4. Symmetric difference
    if len(src_indices) > 0 or len(tgt_indices) > 0:
        # Extract points in symmetric difference
        src_diff = src_pc_transformed[src_indices]
        tgt_diff = tgt_pc[tgt_indices]

        # Combine the points
        sym_diff_pc = torch.cat([src_diff, tgt_diff], dim=0)

        # Create colors for symmetric difference (red for source, blue for target)
        src_colors = torch.ones((len(src_indices), 3), device=src_diff.device)
        src_colors[:, 0] = 1.0  # Red for source
        src_colors[:, 1] = 0.0
        src_colors[:, 2] = 0.0

        tgt_colors = torch.ones((len(tgt_indices), 3), device=tgt_diff.device)
        tgt_colors[:, 0] = 0.0  # Blue for target
        tgt_colors[:, 1] = 0.0
        tgt_colors[:, 2] = 1.0

        sym_diff_colors = torch.cat([src_colors, tgt_colors], dim=0)

        figures.append(create_3d_figure(
            sym_diff_pc,
            colors=sym_diff_colors,
            title="Symmetric Difference",
            point_size=point_size,
            opacity=point_opacity,
            camera_state=camera_state,
            colorscale=None  # Use custom colors
        ))
    else:
        # If no symmetric difference, show empty point cloud
        figures.append(create_3d_figure(
            torch.zeros((1, 3), device=src_pc.device),
            title="Symmetric Difference (Empty)",
            point_size=point_size,
            opacity=point_opacity,
            camera_state=camera_state
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

    # Create a grid layout for the four figures
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
    radius: float = 0.05
) -> html.Div:
    """Display a batched point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        radius: Radius for computing symmetric difference

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

        # For top level (level 0), show union and symmetric difference
        if level == 0:
            # Source point cloud
            figures.append(create_3d_figure(
                src_points,
                title=f"Source Point Cloud (Level {level})",
                point_size=point_size,
                opacity=point_opacity,
                camera_state=camera_state,
            ))

            # Target point cloud
            figures.append(create_3d_figure(
                tgt_points,
                title=f"Target Point Cloud (Level {level})",
                point_size=point_size,
                opacity=point_opacity,
                camera_state=camera_state,
            ))

            # Union of source and target
            union_pc = torch.cat([src_points, tgt_points], dim=0)
            src_colors = torch.ones((len(src_points), 3), device=src_points.device)
            src_colors[:, 0] = 1.0  # Red for source
            tgt_colors = torch.ones((len(tgt_points), 3), device=tgt_points.device)
            tgt_colors[:, 2] = 1.0  # Blue for target
            union_colors = torch.cat([src_colors, tgt_colors], dim=0)

            figures.append(create_3d_figure(
                union_pc,
                colors=union_colors,
                title=f"Union (Level {level})",
                point_size=point_size,
                opacity=point_opacity,
                camera_state=camera_state,
                colorscale=None
            ))

            # Symmetric difference
            src_indices, tgt_indices = pc_symmetric_difference(src_points, tgt_points, radius)
            if len(src_indices) > 0 or len(tgt_indices) > 0:
                src_diff = src_points[src_indices]
                tgt_diff = tgt_points[tgt_indices]
                sym_diff_pc = torch.cat([src_diff, tgt_diff], dim=0)
                src_colors = torch.ones((len(src_indices), 3), device=src_diff.device)
                src_colors[:, 0] = 1.0
                tgt_colors = torch.ones((len(tgt_indices), 3), device=tgt_diff.device)
                tgt_colors[:, 2] = 1.0
                sym_diff_colors = torch.cat([src_colors, tgt_colors], dim=0)

                figures.append(create_3d_figure(
                    sym_diff_pc,
                    colors=sym_diff_colors,
                    title=f"Symmetric Difference (Level {level})",
                    point_size=point_size,
                    opacity=point_opacity,
                    camera_state=camera_state,
                    colorscale=None
                ))
            else:
                figures.append(create_3d_figure(
                    torch.zeros((1, 3), device=src_points.device),
                    title=f"Symmetric Difference (Empty) (Level {level})",
                    point_size=point_size,
                    opacity=point_opacity,
                    camera_state=camera_state
                ))
        else:
            # For lower levels, only show source and target
            figures.append(create_3d_figure(
                src_points,
                title=f"Source Point Cloud (Level {level})",
                point_size=point_size,
                opacity=point_opacity,
                camera_state=camera_state
            ))

            figures.append(create_3d_figure(
                tgt_points,
                title=f"Target Point Cloud (Level {level})",
                point_size=point_size,
                opacity=point_opacity,
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
    radius: float = 0.05
) -> html.Div:
    """Display a point cloud registration datapoint.

    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        camera_state: Optional dictionary containing camera position state
        radius: Radius for computing symmetric difference

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
            radius=radius,
        )
    else:
        return display_pcr_datapoint_single(
            datapoint,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            radius=radius,
        )
