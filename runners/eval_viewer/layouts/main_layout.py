from typing import List
import dash
from dash import html, dcc


def create_controls(max_epoch: int, metrics: List[str]) -> html.Div:
    """
    Creates the control panel with epoch slider and metric dropdown.

    Args:
        max_epoch: Maximum epoch index
        metrics: List of available metrics

    Returns:
        controls: HTML div containing the controls
    """
    return html.Div([
        html.Div([
            html.Label("Epoch:"),
            dcc.Slider(
                id='epoch-slider',
                min=0,
                max=max_epoch,
                step=1,
                value=0,
                marks={i: str(i) for i in range(max_epoch + 1)},
            ),
        ], style={'width': '50%', 'display': 'inline-block'}),

        html.Div([
            html.Label("Metric:"),
            dcc.Dropdown(
                id='metric-dropdown',
                options=[{'label': metric, 'value': metric} for metric in sorted(metrics)],
                value=metrics[0] if metrics else None,
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'padding': '20px'})


def create_score_maps_grid(num_runs: int) -> html.Div:
    """
    Creates the grid layout for displaying score maps.

    Args:
        num_runs: Number of runs to display

    Returns:
        grid: HTML div containing the score maps grid
    """
    return html.Div([
        html.Div(id=f'score-map-{i}', style={'width': '50%', 'display': 'inline-block'})
        for i in range(num_runs)
    ], style={'display': 'flex', 'flexWrap': 'wrap'})


def create_layout(max_epoch: int, metrics: List[str], num_runs: int) -> html.Div:
    """
    Creates the main dashboard layout.

    Args:
        max_epoch: Maximum epoch index
        metrics: List of available metrics
        num_runs: Number of runs to display

    Returns:
        layout: HTML div containing the complete layout
    """
    return html.Div([
        html.H1("Evaluation Viewer", style={'textAlign': 'center'}),
        create_controls(max_epoch, metrics),
        create_score_maps_grid(num_runs),
    ])
