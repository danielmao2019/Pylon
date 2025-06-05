from typing import List, Dict
import numpy as np
import dash
from dash import Input, Output, dcc, html
from dash.exceptions import PreventUpdate

from runners.eval_viewer.backend.initialization import LogDirInfo
from runners.eval_viewer.backend.visualization import create_aggregated_scores_plot, create_overlaid_score_map


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


def create_button_grid(score_map: np.ndarray, button_type: str, run_idx: int = None) -> html.Div:
    """Create a button grid from a score map.
    
    Args:
        score_map: Score map array of shape (H, W)
        button_type: Type of button ('overlaid-grid-button' for overlaid, 'individual-grid-button' for individual)
        run_idx: Index of the run (only needed for individual buttons)
        
    Returns:
        Button grid as an HTML div
    """
    side_length = score_map.shape[0]
    n_datapoints = np.count_nonzero(~np.isnan(score_map))
    
    buttons = []
    for row in range(side_length):
        for col in range(side_length):
            idx = row * side_length + col
            if idx >= n_datapoints:
                continue  # Skip padding cells
            value = score_map[row, col]
            if not np.isnan(value):
                color = get_color_for_score(value, np.nanmin(score_map), np.nanmax(score_map))
                button_id = {'type': button_type, 'index': f'{run_idx}-{row}-{col}' if run_idx is not None else f'{row}-{col}'}
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
        Output('overlaid-button-grid', 'children'),
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
        return create_button_grid(overlaid_score_map, 'overlaid-grid-button')

    outputs = [Output(f'individual-button-grid-{i}', 'children') for i in range(len(log_dir_infos))]
    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_individual_score_maps(epoch: int, metric_name: str):
        if metric_name is None or epoch is None:
            raise PreventUpdate

        metric_idx = metric_names.index(metric_name)
        figures = []
        for i, info in enumerate(log_dir_infos.values()):
            score_map = info.score_map[epoch, metric_idx]
            button_grid = create_button_grid(score_map, 'individual-grid-button', run_idx=i)
            figures.append(button_grid)
        return figures
