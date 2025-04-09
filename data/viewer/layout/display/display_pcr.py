"""UI components for displaying point cloud registration dataset items."""
from typing import Dict, Optional, Union, Any, List, Tuple
import numpy as np
import torch
from dash import dcc, html
import plotly.graph_objects as go
from data.viewer.utils.dataset_utils import format_value
from scipy.spatial import cKDTree


def display_pcr_datapoint(
    datapoint: Dict[str, Any],
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
    radius: float = 0.05
) -> html.Div:
    """Display a point cloud registration datapoint with all relevant information.

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

    # Extract transform if available
    transform = datapoint['labels'].get('transform')
    if transform is None:
        transform = torch.eye(4)  # Default to identity transform if not provided

    # Apply transform to source point cloud
    src_pc_transformed = apply_transform(src_pc, transform)

    # Compute symmetric difference
    src_indices, tgt_indices = compute_symmetric_difference_indices(src_pc_transformed, tgt_pc, radius)

    # Create the four point cloud views
    figures = []

    # 1. Source point cloud (original)
    figures.append(create_3d_figure(
        src_pc,
        title="Source Point Cloud",
        point_size=point_size,
        opacity=point_opacity,
        camera_state=camera_state
    ))

    # 2. Target point cloud
    figures.append(create_3d_figure(
        tgt_pc,
        title="Target Point Cloud",
        point_size=point_size,
        opacity=point_opacity,
        camera_state=camera_state
    ))

    # 3. Union of transformed source and target
    union_pc = torch.cat([src_pc_transformed, tgt_pc], dim=0)
    figures.append(create_3d_figure(
        union_pc,
        title="Union (Transformed Source + Target)",
        point_size=point_size,
        opacity=point_opacity,
        camera_state=camera_state
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
            html.H4("Transform Matrix:"),
            html.Pre(format_value(transform))
        ], style={'margin-top': '20px'})
    ])


def compute_symmetric_difference_indices(src_pc, tgt_pc, radius):
    """
    Compute the indices of points in the symmetric difference between two point clouds using KDTree.

    Args:
        src_pc: torch.Tensor of shape (M, 3) - source point cloud coordinates
        tgt_pc: torch.Tensor of shape (N, 3) - target point cloud coordinates
        radius: float - radius for neighborhood search

    Returns:
        Tuple of (src_indices, tgt_indices) - indices of points in the symmetric difference
    """
    # Input validation
    assert isinstance(src_pc, torch.Tensor), "src_pc must be a torch.Tensor"
    assert isinstance(tgt_pc, torch.Tensor), "tgt_pc must be a torch.Tensor"
    assert isinstance(radius, (int, float)), "radius must be a numeric value"
    assert radius > 0, "radius must be positive"

    # Check tensor dimensions
    assert src_pc.dim() == 2, f"src_pc must be 2D tensor, got {src_pc.dim()}D"
    assert tgt_pc.dim() == 2, f"tgt_pc must be 2D tensor, got {tgt_pc.dim()}D"
    assert src_pc.shape[1] == 3, f"src_pc must have shape (M, 3), got {src_pc.shape}"
    assert tgt_pc.shape[1] == 3, f"tgt_pc must have shape (N, 3), got {tgt_pc.shape}"

    # Check for empty point clouds
    if src_pc.shape[0] == 0 or tgt_pc.shape[0] == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Check for NaN or Inf values
    if torch.isnan(src_pc).any() or torch.isnan(tgt_pc).any() or torch.isinf(src_pc).any() or torch.isinf(tgt_pc).any():
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    # Convert to numpy for scipy's cKDTree
    src_np = src_pc.cpu().numpy()
    tgt_np = tgt_pc.cpu().numpy()

    # Build KDTree for target point cloud
    tgt_tree = cKDTree(tgt_np)

    # Find points in src_pc that are not close to any point in tgt_pc
    # Query the KDTree for all points in src_pc
    distances, _ = tgt_tree.query(src_np, k=1)  # k=1 to find the nearest neighbor
    src_diff_mask = distances > radius
    src_indices = torch.where(torch.from_numpy(src_diff_mask))[0]

    # Build KDTree for source point cloud
    src_tree = cKDTree(src_np)

    # Find points in tgt_pc that are not close to any point in src_pc
    distances, _ = src_tree.query(tgt_np, k=1)  # k=1 to find the nearest neighbor
    tgt_diff_mask = distances > radius
    tgt_indices = torch.where(torch.from_numpy(tgt_diff_mask))[0]

    return src_indices, tgt_indices


def apply_transform(points, transform):
    """
    Apply a transformation to a point cloud.

    Args:
        points: torch.Tensor of shape (N, 3) - point cloud coordinates
        transform: Union[List[List[Union[int, float]]], numpy.ndarray, torch.Tensor] - transformation matrix

    Returns:
        torch.Tensor of shape (N, 3) - transformed point cloud coordinates
    """
    # Convert transform to torch.Tensor if it's not already
    if isinstance(transform, list):
        transform = torch.tensor(transform, dtype=torch.float32)
    elif isinstance(transform, np.ndarray):
        transform = torch.tensor(transform, dtype=torch.float32)

    # Ensure transform is a 4x4 matrix
    assert transform.shape == (4, 4), f"Transform must be a 4x4 matrix, got {transform.shape}"

    # Extract rotation and translation
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    # Apply transformation: R * points + t
    transformed_points = torch.matmul(points, rotation.t()) + translation

    return transformed_points


def tensor_to_point_cloud(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a tensor to a numpy array for point cloud visualization."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def create_3d_figure(
    pc_data: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    colorscale: Optional[str] = 'Viridis',
    point_size: float = 2,
    opacity: float = 0.8,
    colorbar_title: str = "Class",
    camera_state: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """Create a 3D figure for point cloud visualization."""
    # Convert to numpy if needed
    pc_np = tensor_to_point_cloud(pc_data)

    # Create the figure
    fig = go.Figure()

    # Add the point cloud scatter plot
    scatter_kwargs = {
        'x': pc_np[:, 0],
        'y': pc_np[:, 1],
        'z': pc_np[:, 2],
        'mode': 'markers',
        'marker': {
            'size': point_size,
            'opacity': opacity,
        },
        'name': title
    }

    # Add colors if provided
    if colors is not None:
        colors_np = tensor_to_point_cloud(colors)

        # Check if colors are RGB (3 channels) or single values
        if colors_np.shape[1] == 3:
            # RGB colors
            scatter_kwargs['marker']['color'] = colors_np
        else:
            # Single values for colorscale
            scatter_kwargs['marker']['color'] = colors_np.flatten()
            scatter_kwargs['marker']['colorscale'] = colorscale
            scatter_kwargs['marker']['colorbar'] = {'title': colorbar_title}

    fig.add_trace(go.Scatter3d(**scatter_kwargs))

    # Set the camera state if provided
    if camera_state:
        fig.update_layout(
            scene_camera=camera_state
        )

    # Set the layout
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )

    return fig
