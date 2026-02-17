"""Camera display utilities for visualizing camera poses."""

import numpy as np
import plotly.graph_objects as go

from data.structures.three_d.camera.camera import Camera
from data.structures.three_d.camera.camera_vis import camera_vis


def add_camera_to_fig(
    fig: go.Figure,
    camera: Camera,
    axis_length: float = 4.0,
    frustum_depth: float = 8.0,
) -> go.Figure:
    # Input validations
    assert isinstance(fig, go.Figure), f"{type(fig)=}"
    assert isinstance(camera, Camera), f"{type(camera)=}"
    assert isinstance(axis_length, (int, float)), f"{type(axis_length)=}"
    assert isinstance(frustum_depth, (int, float)), f"{type(frustum_depth)=}"

    vis = camera_vis(
        camera=camera,
        axis_length=axis_length,
        frustum_depth=frustum_depth,
    )

    camera_pos = vis["center"].cpu().numpy()
    fig.add_trace(
        go.Scatter3d(
            x=[camera_pos[0]],
            y=[camera_pos[1]],
            z=[camera_pos[2]],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=4,
                color="black",
                line=dict(color="white", width=2),
            ),
            name="Camera",
            showlegend=False,
        )
    )

    for axis in vis["axes"]:
        start = axis["start"].cpu().numpy()
        end = axis["end"].cpu().numpy()
        color = axis["color"].cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=_rgb_to_hex(color), width=4),
                showlegend=False,
            )
        )

    for line in vis["frustum_lines"]:
        start = line["start"].cpu().numpy()
        end = line["end"].cpu().numpy()
        color = line["color"].cpu().numpy()
        fig.add_trace(
            go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode="lines",
                line=dict(color=_rgb_to_hex(color), width=2),
                showlegend=False,
            )
        )

    points = np.array(
        [line["start"].cpu().numpy() for line in vis["axes"]]
        + [line["end"].cpu().numpy() for line in vis["axes"]]
    )
    frustum_points = np.array(
        [seg["start"].cpu().numpy() for seg in vis["frustum_lines"]]
        + [seg["end"].cpu().numpy() for seg in vis["frustum_lines"]]
    )
    all_points = np.concatenate([points, frustum_points], axis=0)
    _update_fig_ranges_with_points(fig=fig, points=all_points)

    return fig


def _update_fig_ranges_with_points(fig: go.Figure, points: np.ndarray) -> go.Figure:
    # Input validations
    assert isinstance(fig, go.Figure), f"{type(fig)=}"
    assert isinstance(points, np.ndarray), f"{type(points)=}"
    assert points.ndim == 2, f"{points.shape=}"
    assert points.shape[1] == 3, f"{points.shape=}"
    assert points.size > 0, f"{points.shape=}"

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    current_scene = fig.layout.scene
    if current_scene is not None:
        current_x_range = getattr(current_scene.xaxis, "range", [x_min, x_max])
        current_y_range = getattr(current_scene.yaxis, "range", [y_min, y_max])
        current_z_range = getattr(current_scene.zaxis, "range", [z_min, z_max])
    else:
        current_x_range = [x_min, x_max]
        current_y_range = [y_min, y_max]
        current_z_range = [z_min, z_max]

    new_x_range = [min(current_x_range[0], x_min), max(current_x_range[1], x_max)]
    new_y_range = [min(current_y_range[0], y_min), max(current_y_range[1], y_max)]
    new_z_range = [min(current_z_range[0], z_min), max(current_z_range[1], z_max)]

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=new_x_range),
            yaxis=dict(range=new_y_range),
            zaxis=dict(range=new_z_range),
        )
    )

    return fig


def _rgb_to_hex(color: np.ndarray) -> str:
    # Input validations
    assert isinstance(color, np.ndarray), f"{type(color)=}"
    assert color.ndim == 1, f"{color.shape=}"
    assert color.shape[0] == 3, f"{color.shape=}"
    assert np.isfinite(color).all(), f"{color=}"
    assert np.all(color >= 0.0), f"{color=}"
    assert np.all(color <= 1.0), f"{color=}"

    r, g, b = (color * 255).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"
