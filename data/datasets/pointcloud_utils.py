import numpy as np
import plotly.graph_objects as go
import torch

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