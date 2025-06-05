from typing import List
from dash import html, dcc
from runners.eval_viewer.layouts.datapoint_viewer import create_datapoint_viewer_layout


def create_controls(max_epoch: int, metric_names: List[str]) -> html.Div:
    """
    Creates the control panel with epoch slider and metric dropdown.

    Args:
        max_epoch: Maximum epoch index
        metric_names: List of available metrics

    Returns:
        controls: HTML div containing the controls
    """
    assert isinstance(max_epoch, int), f"{type(max_epoch)=}"
    assert isinstance(metric_names, list), f"{type(metric_names)=}"
    assert len(metric_names) > 0, f"{metric_names=}"
    assert all(isinstance(metric, str) for metric in metric_names), f"{metric_names=}"
    assert all(metric_names), f"{metric_names=}"

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
                options=[{'label': metric, 'value': metric} for metric in sorted(metric_names)],
                value=metric_names[0] if metric_names else None,
            ),
        ], style={'width': '50%', 'display': 'inline-block', 'float': 'right'}),
    ], style={'padding': '20px'})


def create_aggregated_scores_plot() -> html.Div:
    """
    Creates the plot for visualizing aggregated scores over epochs.

    Returns:
        plot: HTML div containing the aggregated scores plot
    """
    return html.Div([
        html.H2("Aggregated Scores Over Time", style={'textAlign': 'center'}),
        html.Div(id='aggregated-scores-plot', style={'width': '100%'})
    ], style={'marginTop': '20px'})


def create_score_maps_grid(num_runs: int) -> html.Div:
    """Combines the three sections into the main grid layout."""
    return html.Div([
        create_overlaid_score_map_layout(),
        create_individual_score_maps_layout(num_runs),
    ])


def create_individual_score_maps_layout(num_runs: int) -> html.Div:
    """Section for individual score maps."""
    return html.Div([
        html.H2("Individual Score Maps", style={'textAlign': 'center'}),
        html.Div([
            html.Div(id=f'individual-score-map-{i}', style={'width': '50%', 'display': 'inline-block'})
            for i in range(num_runs)
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    ], style={'marginTop': '20px'})


def create_overlaid_score_map_layout() -> html.Div:
    """Section for the button grid (overlaid score map)."""
    return html.Div([
        html.H2("Common Failure Cases", style={'textAlign': 'center'}),
        html.Div(id='overlaid-button-grid', style={
            'display': 'grid',
            'gridTemplateColumns': 'repeat(auto-fill, minmax(20px, 1fr))',
            'gap': '1px',
            'width': '100%',
            'maxWidth': '800px',
            'margin': '0 auto'
        })
    ], style={'marginTop': '20px'})


def create_datapoint_display_section() -> html.Div:
    """Section for the selected datapoint display."""
    return html.Div([
        html.H3("Selected Datapoint", style={'textAlign': 'center'}),
        html.Div(id='datapoint-display', style={'width': '100%'})
    ], style={'marginTop': '20px'})


def create_layout(max_epoch: int, metric_names: List[str], num_runs: int) -> html.Div:
    """
    Creates the main dashboard layout.

    Args:
        max_epoch: Maximum epoch index
        metric_names: List of available metrics
        num_runs: Number of runs to display

    Returns:
        layout: HTML div containing the complete layout
    """
    return html.Div([
        html.H1("Evaluation Viewer", style={'textAlign': 'center'}),
        html.Div([
            # Left column: Score maps and controls
            html.Div([
                create_controls(max_epoch, metric_names),
                create_aggregated_scores_plot(),
                create_score_maps_grid(num_runs),
            ], style={
                'width': '60%',
                'float': 'left',
                'height': 'calc(100vh - 100px)',
                'overflowY': 'auto',
                'padding': '20px',
                'boxSizing': 'border-box'
            }),

            # Right column: Datapoint viewer
            html.Div([
                create_datapoint_display_section(),
            ], style={
                'width': '40%',
                'float': 'right',
                'height': 'calc(100vh - 100px)',
                'overflowY': 'auto',
                'padding': '20px',
                'boxSizing': 'border-box',
                'borderLeft': '1px solid #ddd'
            })
        ], style={
            'display': 'flex',
            'width': '100%',
            'height': 'calc(100vh - 100px)',
            'position': 'relative'
        })
    ])
