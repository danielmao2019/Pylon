from typing import List
import torch
import numpy as np
import plotly.graph_objects as go
from utils.input_checks.check_camera import check_camera_extrinsics


def add_camera_to_fig(fig, camera_extrinsics: torch.Tensor, convention: str) -> go.Figure:
    """Add camera visualization to a 3D plotly figure.
    
    Args:
        fig: Plotly figure to add camera to
        camera_extrinsics: [4, 4] camera extrinsic matrix (camera-to-world transform)
        convention: Camera convention ('opencv', 'opengl', or 'standard')
        
    Returns:
        Updated plotly figure with camera visualization
    """
    check_camera_extrinsics(camera_extrinsics)
    # 1. Camera position as diamond marker
    camera_pos = camera_extrinsics[:3, 3].cpu().numpy()
    fig.add_trace(go.Scatter3d(
        x=[camera_pos[0]],
        y=[camera_pos[1]], 
        z=[camera_pos[2]],
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=4,
            color='black',
            line=dict(color='white', width=2)
        ),
        name='Camera',
        showlegend=False
    ))
    
    # 2. Camera local frame visualization - X red, Y green, Z blue
    fig = _add_camera_axes_vis(fig, camera_extrinsics, convention)
    
    # 3. Camera frustum visualization (in gold color)
    fig = _add_camera_frustum_vis(fig, camera_extrinsics, convention)
    
    return fig


def _update_fig_ranges_with_points(fig, points: np.ndarray) -> go.Figure:
    """Update figure axis ranges to include given points.
    
    Args:
        fig: Plotly figure to update
        points: Array of shape (N, 3) containing points to include
        
    Returns:
        Updated figure with expanded axis ranges
    """
    # Calculate point bounds
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max() 
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # Get current figure ranges - handle Plotly Scene object attributes
    current_scene = fig.layout.scene
    if current_scene is not None:
        # Extract current ranges from Plotly Scene object
        current_x_range = getattr(current_scene.xaxis, 'range', None) or [x_min, x_max]
        current_y_range = getattr(current_scene.yaxis, 'range', None) or [y_min, y_max]
        current_z_range = getattr(current_scene.zaxis, 'range', None) or [z_min, z_max]
    else:
        # No existing scene, use point bounds as initial ranges
        current_x_range = [x_min, x_max]
        current_y_range = [y_min, y_max]
        current_z_range = [z_min, z_max]
    
    # Expand ranges to include new points
    new_x_range = [min(current_x_range[0], x_min), max(current_x_range[1], x_max)]
    new_y_range = [min(current_y_range[0], y_min), max(current_y_range[1], y_max)]
    new_z_range = [min(current_z_range[0], z_min), max(current_z_range[1], z_max)]
    
    # Update figure layout with expanded ranges
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=new_x_range),
            yaxis=dict(range=new_y_range),
            zaxis=dict(range=new_z_range)
        )
    )
    
    return fig


def _add_camera_axes_vis(fig, camera_extrinsics: torch.Tensor, convention: str) -> go.Figure:
    """Add camera axes visualization - ALWAYS red=right, green=forward, blue=up."""
    camera_pos = camera_extrinsics[:3, 3].cpu().numpy()
    rotation_matrix = camera_extrinsics[:3, :3].cpu().numpy()
    scale = 4
    
    # ALWAYS use standard RGB axis colors: red=right, green=forward, blue=up
    # Map camera coordinate system to standard world axes based on convention
    if convention == 'opencv':
        # OpenCV: X-right, Y-down, Z-forward
        right_axis = rotation_matrix[:, 0] * scale    # Red - right (X)
        forward_axis = rotation_matrix[:, 2] * scale  # Green - forward (Z)
        up_axis = -rotation_matrix[:, 1] * scale     # Blue - up (-Y, since Y is down)
    elif convention == 'opengl':
        # OpenGL: X-right, Y-up, Z-backward
        right_axis = rotation_matrix[:, 0] * scale    # Red - right (X)
        forward_axis = -rotation_matrix[:, 2] * scale # Green - forward (-Z, since Z is backward)
        up_axis = rotation_matrix[:, 1] * scale      # Blue - up (Y)
    elif convention == 'standard':
        # Standard: X-right, Y-forward, Z-up
        right_axis = rotation_matrix[:, 0] * scale    # Red - right (X)
        forward_axis = rotation_matrix[:, 1] * scale  # Green - forward (Y)
        up_axis = rotation_matrix[:, 2] * scale      # Blue - up (Z)
    else:
        raise ValueError(f"Unknown convention: {convention}")
    
    # Right axis (red)
    fig.add_trace(go.Scatter3d(
        x=[camera_pos[0], camera_pos[0] + right_axis[0]],
        y=[camera_pos[1], camera_pos[1] + right_axis[1]],
        z=[camera_pos[2], camera_pos[2] + right_axis[2]],
        mode='lines',
        line=dict(color='red', width=4),
        showlegend=False
    ))
    
    # Forward axis (green)
    fig.add_trace(go.Scatter3d(
        x=[camera_pos[0], camera_pos[0] + forward_axis[0]],
        y=[camera_pos[1], camera_pos[1] + forward_axis[1]],
        z=[camera_pos[2], camera_pos[2] + forward_axis[2]],
        mode='lines',
        line=dict(color='green', width=4),
        showlegend=False
    ))
    
    # Up axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[camera_pos[0], camera_pos[0] + up_axis[0]],
        y=[camera_pos[1], camera_pos[1] + up_axis[1]], 
        z=[camera_pos[2], camera_pos[2] + up_axis[2]],
        mode='lines',
        line=dict(color='blue', width=4),
        showlegend=False
    ))
    
    # Update figure ranges to include camera position and all axis endpoints
    axis_points = np.array([
        camera_pos,
        camera_pos + right_axis,
        camera_pos + forward_axis,
        camera_pos + up_axis
    ])
    fig = _update_fig_ranges_with_points(fig, axis_points)
    
    return fig


def _add_camera_frustum_vis(fig, camera_extrinsics: torch.Tensor, convention: str) -> go.Figure:
    """Add camera frustum visualization in gold color."""
    frustum_depth = 8
    frustum_points = _get_camera_frustum_points(camera_extrinsics, frustum_depth, convention)
    
    # Add frustum wireframe
    for line_start, line_end in _get_frustum_lines(frustum_points):
        fig.add_trace(go.Scatter3d(
            x=[line_start[0], line_end[0]],
            y=[line_start[1], line_end[1]],
            z=[line_start[2], line_end[2]],
            mode='lines',
            line=dict(color='gold', width=2),
            showlegend=False
        ))
    
    # Update figure ranges to include all frustum points
    fig = _update_fig_ranges_with_points(fig, frustum_points)
    
    return fig


def _get_camera_frustum_points(camera_extrinsics: torch.Tensor, depth: float, convention: str) -> np.ndarray:
    """Get 3D points for camera frustum visualization."""
    camera_center = camera_extrinsics[:3, 3].cpu().numpy()
    
    # Define frustum corners in camera coordinates based on convention
    if convention == 'opencv':
        # OpenCV: X-right, Y-down, Z-forward
        corners_cam = np.array([
            [-0.5, -0.5, 1.0],  # Top-left
            [ 0.5, -0.5, 1.0],  # Top-right
            [ 0.5,  0.5, 1.0],  # Bottom-right
            [-0.5,  0.5, 1.0]   # Bottom-left
        ]) * depth
    elif convention == 'opengl':
        # OpenGL: X-right, Y-up, Z-backward
        corners_cam = np.array([
            [-0.5,  0.5, -1.0],  # Top-left
            [ 0.5,  0.5, -1.0],  # Top-right
            [ 0.5, -0.5, -1.0],  # Bottom-right
            [-0.5, -0.5, -1.0]   # Bottom-left
        ]) * depth
    else:  # standard
        # Standard: X-right, Y-forward, Z-up
        corners_cam = np.array([
            [-0.5,  1.0,  0.5],  # Top-left
            [ 0.5,  1.0,  0.5],  # Top-right
            [ 0.5,  1.0, -0.5],  # Bottom-right
            [-0.5,  1.0, -0.5]   # Bottom-left
        ]) * depth
    
    # Transform to world coordinates
    corners_cam_homo = np.hstack([corners_cam, np.ones((4, 1))])
    camera_extrinsics_np = camera_extrinsics.cpu().numpy()
    world_coords = (camera_extrinsics_np @ corners_cam_homo.T).T
    frustum_corners = world_coords[:, :3]
    
    return np.vstack([camera_center.reshape(1, 3), frustum_corners])


def _get_frustum_lines(frustum_points: np.ndarray) -> List[tuple]:
    """Get line segments for frustum wireframe."""
    center = frustum_points[0]
    corners = frustum_points[1:]
    
    lines = []
    
    # Lines from camera center to each corner
    for corner in corners:
        lines.append((center, corner))
    
    # Rectangle connecting corners
    for i in range(4):
        lines.append((corners[i], corners[(i + 1) % 4]))
    
    return lines
