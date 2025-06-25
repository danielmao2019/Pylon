from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate
import dash
import plotly.graph_objects as go
from runners.eval_viewer.backend.initialization import LogDirInfo
from runners.eval_viewer.backend.visualization import create_overlaid_score_map
from runners.eval_viewer.layout.main_layout import create_button_grid, create_color_bar


def create_aggregated_scores_plot(epoch_scores: List[np.ndarray], log_dirs: List[str], metric_name: str) -> go.Figure:
    """
    Creates a line plot showing aggregated scores over epochs for each run.

    Args:
        epoch_scores: List of dictionaries containing aggregated scores for each epoch
        log_dirs: List of log directory paths
        metric_name: Name of the metric to plot

    Returns:
        fig: Plotly figure object
    """
    assert isinstance(epoch_scores, list)
    assert all(isinstance(scores, np.ndarray) for scores in epoch_scores)

    fig = go.Figure()

    for scores, log_dir in zip(epoch_scores, log_dirs):
        fig.add_trace(go.Scatter(
            x=list(range(len(scores))),
            y=scores,
            name=log_dir.split('/')[-1],
            mode='lines+markers'
        ))

    fig.update_layout(
        title=f"Aggregated {metric_name} Over Time",
        xaxis_title="Epoch",
        yaxis_title="Score",
        showlegend=True,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    return fig


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

        # Get scores for all epochs from all runs, handling different shapes
        metric_idx = metric_names.index(metric_name)
        epoch_scores = []
        for info in log_dir_infos.values():
            if info.runner_type == 'trainer':
                # For trainer: aggregated_scores has shape (N, C)
                epoch_scores.append(info.aggregated_scores[:, metric_idx])
            elif info.runner_type == 'evaluator':
                # For evaluator: aggregated_scores has shape (C,), replicate to show as single point
                epoch_scores.append(np.array([info.aggregated_scores[metric_idx]]))
            else:
                raise ValueError(f"Unknown runner type: {info.runner_type}")

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
        score_maps = []
        for info in log_dir_infos.values():
            if info.runner_type == 'trainer':
                # For trainer: use epoch from slider, score_map has shape (N, C, H, W)
                score_maps.append(info.score_map[epoch, metric_idx])
            elif info.runner_type == 'evaluator':
                # For evaluator: ignore epoch slider, score_map has shape (C, H, W)
                score_maps.append(info.score_map[metric_idx])
            else:
                raise ValueError(f"Unknown runner type: {info.runner_type}")

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

        # Get all score maps for this epoch and metric, handling different shapes
        score_maps = []
        for info in log_dir_infos.values():
            if info.runner_type == 'trainer':
                # For trainer: use epoch from slider, score_map has shape (N, C, H, W)
                score_maps.append(info.score_map[epoch, metric_idx])
            elif info.runner_type == 'evaluator':
                # For evaluator: ignore epoch slider, score_map has shape (C, H, W)
                score_maps.append(info.score_map[metric_idx])
            else:
                raise ValueError(f"Unknown runner type: {info.runner_type}")

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
