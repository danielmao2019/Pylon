"""UI components for displaying dataset items."""
from dash import dcc, html
import plotly.graph_objects as go
import torch
import numpy as np
from data.viewer.utils.dataset_utils import format_value
from data.viewer.utils.pointcloud_utils import create_synchronized_point_cloud_figures


def get_point_cloud_stats(pc, change_map=None, class_names=None):
    """Get statistical information about a point cloud.

    Args:
        pc: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        Dictionary with point cloud statistics
    """
    if not isinstance(pc, torch.Tensor):
        return {}

    try:
        # Basic stats
        pc_np = pc.detach().cpu().numpy()
        stats = {
            "Total Points": len(pc_np),
            "Dimensions": pc_np.shape[1],
            "X Range": f"[{pc_np[:, 0].min():.2f}, {pc_np[:, 0].max():.2f}]",
            "Y Range": f"[{pc_np[:, 1].min():.2f}, {pc_np[:, 1].max():.2f}]",
            "Z Range": f"[{pc_np[:, 2].min():.2f}, {pc_np[:, 2].max():.2f}]",
            "Center": f"[{pc_np[:, 0].mean():.2f}, {pc_np[:, 1].mean():.2f}, {pc_np[:, 2].mean():.2f}]",
        }

        # Add class distribution if change_map is provided
        if change_map is not None:
            unique_classes, class_counts = torch.unique(change_map, return_counts=True)

            # Convert to numpy for display
            unique_classes = unique_classes.cpu().numpy()
            class_counts = class_counts.cpu().numpy()

            # Calculate distribution
            total_points = change_map.numel()
            class_distribution = []

            for cls, count in zip(unique_classes, class_counts):
                percentage = (count / total_points) * 100
                cls_key = cls.item() if hasattr(cls, 'item') else cls

                if class_names and cls_key in class_names:
                    class_label = class_names[cls_key]
                    class_distribution.append({
                        "class_id": cls_key,
                        "class_name": class_label,
                        "count": int(count),
                        "percentage": percentage
                    })
                else:
                    class_distribution.append({
                        "class_id": cls_key,
                        "class_name": f"Class {cls_key}",
                        "count": int(count),
                        "percentage": percentage
                    })

            stats["class_distribution"] = class_distribution

        return stats
    except Exception as e:
        print(f"Error calculating point cloud stats: {e}")
        return {"error": str(e)}


def display_3d_datapoint(datapoint, point_size=2, point_opacity=0.8, class_names=None):
    """
    Display a 3D point cloud datapoint with all relevant information.
    
    Args:
        datapoint: Dictionary containing inputs, labels, and meta_info
        point_size: Size of points in visualization
        point_opacity: Opacity of points in visualization
        class_names: Optional dictionary mapping class indices to names
        
    Returns:
        html.Div containing the visualization
    """
    # Check if the inputs have the expected structure
    pc_1 = datapoint['inputs'].get('pc_1')
    pc_2 = datapoint['inputs'].get('pc_2')
    change_map = datapoint['labels'].get('change_map')
    
    # Verify that all required data is present
    error_messages = []
    
    if pc_1 is None:
        error_messages.append("Point Cloud 1 (pc_1) is missing")
    elif not isinstance(pc_1, torch.Tensor):
        error_messages.append(f"Point Cloud 1 (pc_1) has unexpected type: {type(pc_1).__name__}")
        
    if pc_2 is None:
        error_messages.append("Point Cloud 2 (pc_2) is missing")
    elif not isinstance(pc_2, torch.Tensor):
        error_messages.append(f"Point Cloud 2 (pc_2) has unexpected type: {type(pc_2).__name__}")
        
    if change_map is None:
        error_messages.append("Change map is missing")
    elif not isinstance(change_map, torch.Tensor):
        error_messages.append(f"Change map has unexpected type: {type(change_map).__name__}")
    
    # If any errors were found, display them
    if error_messages:
        return html.Div([
            html.H3("Error Displaying 3D Point Cloud Data", style={'color': 'red'}),
            html.P("The dataset structure doesn't match the expected format:"),
            html.Ul([html.Li(msg) for msg in error_messages]),
            html.P("Datapoint structure:"),
            html.Div([
                html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
                html.Pre(f"Meta info: {format_value(datapoint.get('meta_info', {}))}")
            ], style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'})
        ])

    # Prepare the point clouds for visualization
    try:
        # Get stats for point clouds
        pc_1_stats = get_point_cloud_stats(pc_1, class_names=class_names)
        pc_2_stats = get_point_cloud_stats(pc_2, class_names=class_names)
        change_stats = get_point_cloud_stats(pc_1, change_map, class_names=class_names)
        
        # Create figures for point clouds
        point_clouds = [pc_1, pc_2]
        colors = [None, None]
        
        # For change map visualization, we'll use pc_1 with colors from change_map
        if change_map is not None:
            point_clouds.append(pc_1)
            colors.append(change_map.float())  # Convert to float for proper coloring
        
        titles = ["Point Cloud 1", "Point Cloud 2", "Change Map"]
        
        # Create synchronized 3D views
        figures = create_synchronized_point_cloud_figures(
            point_clouds, 
            colors=colors,
            titles=titles,
            point_sizes=[point_size] * len(point_clouds),
            opacities=[point_opacity] * len(point_clouds),
            colorscales=['Viridis', 'Viridis', 'Reds']
        )
        
        # Extract metadata
        meta_info = datapoint.get('meta_info', {})
        meta_display = []
        if meta_info:
            meta_display = [
                html.H4("Metadata:"),
                html.Pre(format_value(meta_info), 
                      style={'background-color': '#f0f0f0', 'padding': '10px', 'max-height': '200px', 
                             'overflow-y': 'auto', 'border-radius': '5px'})
            ]
        
        # Compile the complete display
        return html.Div([
            # Point cloud displays
            html.Div([
                html.Div([
                    dcc.Graph(figure=figures[0])
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=figures[1])
                ], style={'width': '33%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(figure=figures[2] if len(figures) > 2 else {})
                ], style={'width': '33%', 'display': 'inline-block'}),
            ]),
            
            # Info section
            html.Div([
                # Point cloud statistics
                html.Div([
                    html.Div([
                        html.H4("Point Cloud 1 Statistics:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in pc_1_stats.items()])
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    
                    html.Div([
                        html.H4("Point Cloud 2 Statistics:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in pc_2_stats.items()])
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    
                    html.Div([
                        html.H4("Change Statistics:"),
                        html.Ul([html.Li(f"{k}: {v}") for k, v in change_stats.items()])
                    ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top'}),
                ]),
                
                # Metadata
                html.Div(meta_display, style={'margin-top': '20px'})
            ], style={'margin-top': '20px'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("Error Processing Point Clouds", style={'color': 'red'}),
            html.P(f"An error occurred while processing the point clouds: {str(e)}"),
            html.P("Datapoint structure:"),
            html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
            html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
            html.P(f"Point Cloud 1 shape: {pc_1.shape if pc_1 is not None else 'None'}"),
            html.P(f"Point Cloud 2 shape: {pc_2.shape if pc_2 is not None else 'None'}"),
            html.P(f"Change map shape: {change_map.shape if change_map is not None else 'None'}")
        ])

def tensor_to_point_cloud(tensor):
    """Convert a PyTorch tensor to a displayable point cloud."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor

def create_point_cloud_figure(pc_data, colors=None, title="Point Cloud", colorscale='Viridis',
                             point_size=2, opacity=0.8, colorbar_title="Class"):
    """Create a 3D point cloud visualization figure.

    Args:
        pc_data: Numpy array of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N,) containing color values
        title: Title for the figure
        colorscale: Colorscale to use for the point cloud
        point_size: Size of the points
        opacity: Opacity of the points
        colorbar_title: Title for the colorbar

    Returns:
        Plotly Figure object
    """
    # Make sure pc_data is a numpy array
    pc_data = tensor_to_point_cloud(pc_data)

    # Subsample large point clouds if necessary for better performance
    max_points = 100000  # Adjust based on performance needs
    if pc_data.shape[0] > max_points:
        indices = np.random.choice(pc_data.shape[0], max_points, replace=False)
        pc_data = pc_data[indices]
        if colors is not None:
            colors = tensor_to_point_cloud(colors)[indices]

    # Create figure
    fig = go.Figure()

    # Add point cloud
    if colors is not None:
        colors = tensor_to_point_cloud(colors)
        fig.add_trace(go.Scatter3d(
            x=pc_data[:, 0],
            y=pc_data[:, 1],
            z=pc_data[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=colors,
                colorscale=colorscale,
                opacity=opacity,
                colorbar=dict(
                    title=colorbar_title,
                    thickness=15,
                    len=0.6,
                    x=1.02,  # Position colorbar outside the plot area
                    xanchor="left",
                    xpad=10
                )
            ),
            text=[f"Point {i}<br>Value: {c}" for i, c in enumerate(colors)],
            hoverinfo='text'
        ))
    else:
        fig.add_trace(go.Scatter3d(
            x=pc_data[:, 0],
            y=pc_data[:, 1],
            z=pc_data[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color='steelblue',
                opacity=opacity
            ),
            text=[f"Point {i}" for i in range(len(pc_data))],
            hoverinfo='text'
        ))

    # Calculate bounding box
    x_range = [pc_data[:, 0].min(), pc_data[:, 0].max()]
    y_range = [pc_data[:, 1].min(), pc_data[:, 1].max()]
    z_range = [pc_data[:, 2].min(), pc_data[:, 2].max()]

    # Set layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            xaxis=dict(range=x_range),
            yaxis=dict(range=y_range),
            zaxis=dict(range=z_range)
        ),
        margin=dict(l=0, r=40, b=0, t=40),  # Increased right margin for colorbar
        height=500,
    )

    return fig

def create_synchronized_point_cloud_figures(point_clouds, colors=None, titles=None,
                                          point_sizes=None, opacities=None, colorscales=None):
    """Create multiple 3D point cloud figures with synchronized camera views.

    Args:
        point_clouds: List of point cloud data arrays
        colors: List of color arrays (optional)
        titles: List of titles for each figure
        point_sizes: List of point sizes for each figure
        opacities: List of opacity values for each figure
        colorscales: List of colorscales for each figure

    Returns:
        List of Plotly Figure objects
    """
    if titles is None:
        titles = [f"Point Cloud {i+1}" for i in range(len(point_clouds))]

    if colors is None:
        colors = [None] * len(point_clouds)

    if point_sizes is None:
        point_sizes = [2] * len(point_clouds)

    if opacities is None:
        opacities = [0.8] * len(point_clouds)

    if colorscales is None:
        colorscales = ['Viridis'] * len(point_clouds)

    figs = []
    for i, (pc, color, title, point_size, opacity, colorscale) in enumerate(
            zip(point_clouds, colors, titles, point_sizes, opacities, colorscales)):
        fig = create_point_cloud_figure(
            pc,
            colors=color,
            title=title,
            point_size=point_size,
            opacity=opacity,
            colorscale=colorscale
        )
        figs.append(fig)

    return figs
