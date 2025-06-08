from typing import List, Dict
import numpy as np
import dash
from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate
from concurrent.futures import ThreadPoolExecutor, as_completed

from runners.eval_viewer.backend.initialization import LogDirInfo
from runners.eval_viewer.backend.visualization import create_aggregated_scores_plot, create_overlaid_score_map
from runners.eval_viewer.layouts.main_layout import create_button_grid, create_color_bar


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


def create_grid_and_colorbar(score_map, run_idx, num_datapoints, min_score, max_score):
    """Create a button grid and color bar for a single score map.

    Args:
        score_map: Score map array
        run_idx: Index of the run
        num_datapoints: Number of datapoints
        min_score: Minimum score value
        max_score: Maximum score value

    Returns:
        Tuple of (run_idx, [button_grid, color_bar])
    """
    button_grid = create_button_grid(
        num_datapoints, score_map, 'individual-grid-button',
        run_idx=run_idx, min_score=min_score, max_score=max_score,
    )
    color_bar = create_color_bar(min_score, max_score)
    return run_idx, [button_grid, color_bar]


def register_callbacks(app: dash.Dash, metric_names: List[str], num_datapoints: int, log_dir_infos: Dict[str, LogDirInfo]):
    """
    Registers all callbacks for the app.

    Args:
        app: Dash application instance
        metric_names: List of metric names
        num_datapoints: Number of datapoints in the dataset
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
        button_grid = create_button_grid(
            num_datapoints, overlaid_score_map, 'overlaid-grid-button',
        )
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

        # Create button grids and color bars in parallel
        results = [None] * len(score_maps)  # Pre-allocate list to maintain order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    create_grid_and_colorbar,
                    score_map, i, num_datapoints, min_score, max_score,
                ): i for i, score_map in enumerate(score_maps)
            }
            
            # Collect results in order
            for future in as_completed(future_to_idx):
                run_idx, grid_and_bar = future.result()
                results[run_idx] = grid_and_bar  # Place results in correct position
        
        # Flatten the results list
        return [item for sublist in results for item in sublist]
