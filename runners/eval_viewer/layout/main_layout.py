from typing import List, Optional
import numpy as np
from dash import html, dcc


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


def create_color_bar(min_score: float, max_score: float) -> html.Div:
    """Create a color bar showing the score range.
    
    Args:
        min_score: Minimum score value
        max_score: Maximum score value
        
    Returns:
        Color bar as an HTML div
    """
    return html.Div([
        html.Div([
            html.Div(style={
                'width': '20px',
                'height': '100px',
                'background': 'linear-gradient(to bottom, rgb(255,0,0), rgb(255,255,0), rgb(0,255,0))',
                'marginRight': '10px'
            }),
            html.Div([
                html.Div(f"{max_score:.2f}", style={'marginBottom': '5px'}),
                html.Div(f"{min_score:.2f}")
            ], style={'display': 'flex', 'flexDirection': 'column', 'justifyContent': 'space-between'})
        ], style={'display': 'flex', 'alignItems': 'center'})
    ], style={'marginLeft': '10px'})


def create_button_grid(
    num_datapoints: int,
    score_map: np.ndarray,
    button_type: str,
    run_idx: int = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
) -> html.Div:
    """Create a button grid from a score map.

    Args:
        num_datapoints: Number of datapoints in the dataset
        score_map: Score map array of shape (H, W)
        button_type: Type of button ('overlaid-grid-button' for overlaid, 'individual-grid-button' for individual)
        run_idx: Index of the run (only needed for individual buttons)
        min_score: Global minimum score for color scaling (if None, use local min)
        max_score: Global maximum score for color scaling (if None, use local max)

    Returns:
        Button grid as an HTML div
    """
    side_length = score_map.shape[0]

    # Use global min/max if provided, otherwise use local min/max
    if min_score is None:
        min_score = np.nanmin(score_map)
    if max_score is None:
        max_score = np.nanmax(score_map)

    buttons = []
    for row in range(side_length):
        for col in range(side_length):
            idx = row * side_length + col
            if idx >= num_datapoints:
                # This is a padding position - no button at all
                buttons.append(html.Div(style={
                    'width': '20px',
                    'height': '20px',
                    'padding': '0',
                    'margin': '0',
                }))
                continue

            value = score_map[row, col]
            button_id = {'type': button_type, 'index': f'{run_idx}-{idx}' if run_idx is not None else str(idx)}

            if np.isnan(value):
                # This is a NaN score - show gray button
                button = html.Button(
                    '',
                    id=button_id,
                    style={
                        'width': '20px',
                        'height': '20px',
                        'padding': '0',
                        'margin': '0',
                        'border': 'none',
                        'backgroundColor': '#f0f0f0',  # Light gray for NaN values
                        'cursor': 'not-allowed'  # Show that these buttons are not clickable
                    }
                )
            else:
                # This is a valid score - show colored button
                color = get_color_for_score(value, min_score, max_score)
                button = html.Button(
                    '',
                    id=button_id,
                    style={
                        'width': '20px',
                        'height': '20px',
                        'padding': '0',
                        'margin': '0',
                        'border': 'none',
                        'backgroundColor': color,
                        'cursor': 'pointer'
                    }
                )
            buttons.append(button)

    return html.Div(buttons, style={
        'display': 'grid',
        'gridTemplateColumns': f'repeat({side_length}, 20px)',
        'gap': '1px',
        'width': 'fit-content',
        'margin': '0 auto'
    })


def create_individual_score_maps_layout(run_names: List[str]) -> html.Div:
    """Section for individual score maps.
    
    Args:
        run_names: List of run names to display
    """
    return html.Div([
        html.H2("Individual Score Maps", style={'textAlign': 'center'}),
        html.Div([
            html.Div([
                html.H3(run_name, style={'textAlign': 'center', 'marginBottom': '10px'}),
                html.Div([
                    html.Div(id=f'individual-button-grid-{i}', style={'width': '100%', 'display': 'inline-block'}),
                    html.Div(id=f'individual-color-bar-{i}', style={'display': 'inline-block'})
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'width': '50%', 'display': 'inline-block'})
            for i, run_name in enumerate(run_names)
        ], style={'display': 'flex', 'flexWrap': 'wrap'})
    ], style={'marginTop': '20px'})


def create_overlaid_score_map_layout() -> html.Div:
    """Section for the button grid (overlaid score map)."""
    return html.Div([
        html.H2("Common Failure Cases", style={'textAlign': 'center'}),
        html.Div([
            html.Div(id='overlaid-button-grid', style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fill, minmax(20px, 1fr))',
                'gap': '1px',
                'width': '100%',
                'maxWidth': '800px',
                'margin': '0 auto'
            }),
            html.Div(id='overlaid-color-bar', style={'display': 'inline-block'})
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
    ], style={'marginTop': '20px'})


def create_score_maps_grid(run_names: List[str]) -> html.Div:
    """Combines the three sections into the main grid layout."""
    return html.Div([
        create_overlaid_score_map_layout(),
        create_individual_score_maps_layout(run_names),
    ])


def create_datapoint_display_section() -> html.Div:
    """Section for the selected datapoint display."""
    return html.Div([
        html.H3("Selected Datapoint", style={'textAlign': 'center'}),
        html.Div(id='datapoint-display', style={'width': '100%'})
    ], style={'marginTop': '20px'})


def create_layout(max_epoch: int, metric_names: List[str], run_names: List[str]) -> html.Div:
    """
    Creates the main dashboard layout.

    Args:
        max_epoch: Maximum epoch index
        metric_names: List of available metrics
        run_names: List of run names to display

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
                create_score_maps_grid(run_names),
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
