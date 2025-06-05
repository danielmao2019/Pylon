from typing import List, Dict, Optional
import numpy as np
import dash
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from runners.eval_viewer.backend.initialization import LogDirInfo
from runners.eval_viewer.backend.visualization import create_aggregated_scores_plot, create_overlaid_score_map
from runners.eval_viewer.layouts.main_layout import create_color_bar


def get_color_for_score(score: float, min_score: float, max_score: float) -> str:
    """Convert a score to a color using a red-yellow-green colormap."""
    if np.isnan(score):
        return '#808080'  # Gray for NaN values

    # Normalize score to [0, 1]
    normalized = (score - min_score) / (max_score - min_score)

    # Create color gradient from red (0) to yellow (0.5) to green (1)
    if normalized < 0.5:
        # Red to Yellow
        r = 1.0
        g = normalized * 2
        b = 0.0
    else:
        # Yellow to Green
        r = 2 * (1 - normalized)
        g = 1.0
        b = 0.0

    return f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})'


def create_button_grid(
    score_map: np.ndarray,
    button_type: str,
    run_idx: int = None,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None,
) -> html.Div:
    """Create a button grid from a score map.

    Args:
        score_map: Score map array of shape (H, W)
        button_type: Type of button ('overlaid-grid-button' for overlaid, 'individual-grid-button' for individual)
        run_idx: Index of the run (only needed for individual buttons)
        min_score: Global minimum score for color scaling (if None, use local min)
        max_score: Global maximum score for color scaling (if None, use local max)

    Returns:
        Button grid as an HTML div
    """
    side_length = score_map.shape[0]
    n_datapoints = np.count_nonzero(~np.isnan(score_map))

    # Use global min/max if provided, otherwise use local min/max
    if min_score is None:
        min_score = np.nanmin(score_map)
    if max_score is None:
        max_score = np.nanmax(score_map)

    buttons = []
    for row in range(side_length):
        for col in range(side_length):
            idx = row * side_length + col
            if idx >= n_datapoints:
                # This is a padding position - no button at all
                buttons.append(html.Div(style={
                    'width': '20px',
                    'height': '20px',
                    'padding': '0',
                    'margin': '0',
                }))
                continue

            value = score_map[row, col]
            button_id = {'type': button_type, 'index': f'{run_idx}-{row}-{col}' if run_idx is not None else f'{row}-{col}'}

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


def register_callbacks(app: dash.Dash, metric_names: List[str], log_dir_infos: Dict[str, LogDirInfo]):
    """
    Registers all callbacks for the app.

    Args:
        app: Dash application instance
        metric_names: List of metric names
        log_dir_infos: Dictionary mapping log directory paths to LogDirInfo objects
    """
    @app.callback(
        Output('aggregated-scores-plot', 'children'),
        [Input('metric-dropdown', 'value')]
    )
    def update_aggregated_scores_plot(metric_name: str) -> dcc.Graph:
        """
        Updates the aggregated scores plot based on selected metric.

        Args:
            epoch: Selected epoch
            metric: Selected metric name

        Returns:
            figure: Plotly figure dictionary for the aggregated scores plot
        """
        if metric_name is None:
            raise PreventUpdate

        # Get scores for all epochs from all runs
        metric_idx = metric_names.index(metric_name)
        epoch_scores = [info.aggregated_scores[:, metric_idx] for info in log_dir_infos.values()]

        # Create figure
        fig = create_aggregated_scores_plot(epoch_scores, list(log_dir_infos.keys()), metric_name=metric_name)
        return dcc.Graph(figure=fig)

    @app.callback(
        [Output('overlaid-button-grid', 'children'),
         Output('overlaid-color-bar', 'children')],
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_overlaid_score_map(epoch: int, metric_name: str):
        if metric_name is None or epoch is None:
            raise PreventUpdate

        metric_idx = metric_names.index(metric_name)
        score_maps = [
            info.score_map[epoch, metric_idx]
            for info in log_dir_infos.values()
        ]
        assert len(score_maps) > 0, f"No score maps found for metric {metric_name}"
        overlaid_score_map = create_overlaid_score_map(score_maps)
        button_grid = create_button_grid(overlaid_score_map, 'overlaid-grid-button')
        min_score = np.nanmin(overlaid_score_map)
        max_score = np.nanmax(overlaid_score_map)
        color_bar = create_color_bar(min_score, max_score)
        return button_grid, color_bar

    outputs = []
    for i in range(len(log_dir_infos)):
        outputs.extend([
            Output(f'individual-button-grid-{i}', 'children'),
            Output(f'individual-color-bar-{i}', 'children')
        ])

    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_individual_score_maps(epoch: int, metric_name: str):
        if metric_name is None or epoch is None:
            raise PreventUpdate

        metric_idx = metric_names.index(metric_name)

        # Get all score maps for this epoch and metric
        score_maps = [
            info.score_map[epoch, metric_idx]
            for info in log_dir_infos.values()
        ]

        # Calculate global min/max scores across all individual maps
        min_score = min(np.nanmin(score_map) for score_map in score_maps)
        max_score = max(np.nanmax(score_map) for score_map in score_maps)

        results = []
        for i, score_map in enumerate(score_maps):
            button_grid = create_button_grid(score_map, 'individual-grid-button', run_idx=i, min_score=min_score, max_score=max_score)
            color_bar = create_color_bar(min_score, max_score)
            results.extend([button_grid, color_bar])
        return results
