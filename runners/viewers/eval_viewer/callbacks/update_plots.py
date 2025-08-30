from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import dash
import plotly.graph_objects as go
from runners.viewers.eval_viewer.backend.initialization import LogDirInfo
from runners.viewers.eval_viewer.backend.visualization import create_overlaid_score_map
from runners.viewers.eval_viewer.layout.main_layout import create_button_grid, create_color_bar


def create_aggregated_scores_plot(epoch_scores: List[np.ndarray], log_dirs: List[str], metric_name: str) -> go.Figure:
    """
    Creates a line plot showing aggregated scores over epochs for each run.

    Args:
        epoch_scores: List of numpy arrays containing aggregated scores for each epoch
        log_dirs: List of log directory paths
        metric_name: Name of the metric to plot

    Returns:
        fig: Plotly figure object
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert epoch_scores is not None, "epoch_scores must not be None"
    assert isinstance(epoch_scores, list), f"epoch_scores must be list, got {type(epoch_scores)}"
    assert len(epoch_scores) > 0, f"epoch_scores must not be empty"
    assert all(isinstance(scores, np.ndarray) for scores in epoch_scores), f"All epoch_scores must be numpy arrays"
    
    assert log_dirs is not None, "log_dirs must not be None"
    assert isinstance(log_dirs, list), f"log_dirs must be list, got {type(log_dirs)}"
    assert len(log_dirs) == len(epoch_scores), f"log_dirs length {len(log_dirs)} must match epoch_scores length {len(epoch_scores)}"
    assert all(isinstance(log_dir, str) for log_dir in log_dirs), f"All log_dirs must be strings"
    
    assert metric_name is not None, "metric_name must not be None"
    assert isinstance(metric_name, str), f"metric_name must be str, got {type(metric_name)}"

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
    # Input validation following CLAUDE.md fail-fast patterns
    assert score_map is not None, "score_map must not be None"
    assert isinstance(score_map, np.ndarray), f"score_map must be numpy array, got {type(score_map)}"
    assert score_map.ndim == 2, f"score_map must be 2D, got shape {score_map.shape}"
    
    assert run_idx is not None, "run_idx must not be None"
    assert isinstance(run_idx, int), f"run_idx must be int, got {type(run_idx)}"
    assert run_idx >= 0, f"run_idx must be non-negative, got {run_idx}"
    
    assert num_datapoints is not None, "num_datapoints must not be None"
    assert isinstance(num_datapoints, int), f"num_datapoints must be int, got {type(num_datapoints)}"
    assert num_datapoints > 0, f"num_datapoints must be positive, got {num_datapoints}"
    
    assert min_score is not None, "min_score must not be None"
    assert isinstance(min_score, (int, float)), f"min_score must be numeric, got {type(min_score)}"
    
    assert max_score is not None, "max_score must not be None"
    assert isinstance(max_score, (int, float)), f"max_score must be numeric, got {type(max_score)}"
    assert max_score >= min_score, f"max_score {max_score} must be >= min_score {min_score}"

    button_grid = create_button_grid(
        num_datapoints=num_datapoints, 
        score_map=score_map, 
        button_type='individual-grid-button',
        run_idx=run_idx, 
        min_score=min_score, 
        max_score=max_score,
    )
    color_bar = create_color_bar(min_score=min_score, max_score=max_score)
    return run_idx, [button_grid, color_bar]


def register_callbacks(app: dash.Dash, metric_names: List[str], num_datapoints: int, log_dir_infos: Dict[str, LogDirInfo], per_metric_color_scales: np.ndarray):
    """
    Registers all callbacks for the app.

    Args:
        app: Dash application instance
        metric_names: List of metric names
        num_datapoints: Number of datapoints in the dataset
        log_dir_infos: Dictionary mapping log directory paths to LogDirInfo objects
        per_metric_color_scales: Array of shape (C, 2) with min/max values for each metric
    """
    # Input validation following CLAUDE.md fail-fast patterns
    assert app is not None, "app must not be None"
    assert isinstance(app, dash.Dash), f"app must be dash.Dash instance, got {type(app)}"
    
    assert metric_names is not None, "metric_names must not be None"
    assert isinstance(metric_names, list), f"metric_names must be list, got {type(metric_names)}"
    assert len(metric_names) > 0, f"metric_names must not be empty, got {metric_names}"
    assert all(isinstance(name, str) for name in metric_names), f"All metric names must be strings, got {metric_names}"
    
    assert num_datapoints is not None, "num_datapoints must not be None"
    assert isinstance(num_datapoints, int), f"num_datapoints must be int, got {type(num_datapoints)}"
    assert num_datapoints > 0, f"num_datapoints must be positive, got {num_datapoints}"
    
    assert log_dir_infos is not None, "log_dir_infos must not be None"
    assert isinstance(log_dir_infos, dict), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert len(log_dir_infos) > 0, f"log_dir_infos must not be empty"
    
    assert per_metric_color_scales is not None, "per_metric_color_scales must not be None"
    assert isinstance(per_metric_color_scales, np.ndarray), f"per_metric_color_scales must be numpy array, got {type(per_metric_color_scales)}"
    assert per_metric_color_scales.shape == (len(metric_names), 2), f"per_metric_color_scales shape must be ({len(metric_names)}, 2), got {per_metric_color_scales.shape}"
    @app.callback(
        Output('aggregated-scores-plot', 'children'),
        [Input('metric-dropdown', 'value')]
    )
    def update_aggregated_scores_plot(metric_name: str) -> dcc.Graph:
        """
        Updates the aggregated scores plot based on selected metric.

        Args:
            metric_name: Selected metric name

        Returns:
            figure: Plotly figure dictionary for the aggregated scores plot
        """
        # Handle None values during app initialization
        if metric_name is None:
            raise PreventUpdate
            
        # Input validation following CLAUDE.md fail-fast patterns
        assert isinstance(metric_name, str), f"metric_name must be str, got {type(metric_name)}"
        assert metric_name in metric_names, f"metric_name {metric_name} not found in available metrics: {metric_names}"

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

        # Create figure using kwargs pattern
        fig = create_aggregated_scores_plot(
            epoch_scores=epoch_scores, 
            log_dirs=list(log_dir_infos.keys()), 
            metric_name=metric_name
        )
        return dcc.Graph(figure=fig)

    @app.callback(
        [Output('overlaid-button-grid', 'children'),
         Output('overlaid-color-bar', 'children')],
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value'),
         Input('percentile-slider', 'drag_value'),
         Input('percentile-slider', 'value')]
    )
    def update_overlaid_score_map(epoch: int, metric_name: str, percentile_drag: float, percentile_value: float):
        # Handle None values during app initialization
        if epoch is None or metric_name is None:
            raise PreventUpdate
            
        # Use drag_value if available (while dragging), otherwise use value
        percentile = percentile_drag if percentile_drag is not None else percentile_value
        if percentile is None:
            raise PreventUpdate
        
        # Input validation following CLAUDE.md fail-fast patterns
        assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
        assert epoch >= 0, f"epoch must be non-negative, got {epoch}"
        
        assert isinstance(metric_name, str), f"metric_name must be str, got {type(metric_name)}"
        assert metric_name in metric_names, f"metric_name {metric_name} not found in available metrics: {metric_names}"
        
        assert isinstance(percentile, (int, float)), f"percentile must be numeric, got {type(percentile)}"
        assert 0 <= percentile <= 100, f"percentile must be between 0 and 100, got {percentile}"

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
        overlaid_score_map = create_overlaid_score_map(score_maps=score_maps, percentile=percentile)
        button_grid = create_button_grid(
            num_datapoints=num_datapoints, 
            score_map=overlaid_score_map, 
            button_type='overlaid-grid-button',
        )
        min_score = np.nanmin(overlaid_score_map)
        max_score = np.nanmax(overlaid_score_map)
        color_bar = create_color_bar(min_score=min_score, max_score=max_score)
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
        # Handle None values during app initialization
        if epoch is None or metric_name is None:
            raise PreventUpdate
            
        # Input validation following CLAUDE.md fail-fast patterns
        assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
        assert epoch >= 0, f"epoch must be non-negative, got {epoch}"
        
        assert isinstance(metric_name, str), f"metric_name must be str, got {type(metric_name)}"
        assert metric_name in metric_names, f"metric_name {metric_name} not found in available metrics: {metric_names}"

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

        # Use pre-computed per-metric color scale for this specific metric
        min_score, max_score = per_metric_color_scales[metric_idx]

        # Create button grids and color bars in parallel
        results = [None] * len(score_maps)  # Pre-allocate list to maintain order
        with ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(
                    create_grid_and_colorbar,
                    score_map=score_map, 
                    run_idx=i, 
                    num_datapoints=num_datapoints, 
                    min_score=min_score, 
                    max_score=max_score,
                ): i for i, score_map in enumerate(score_maps)
            }

            # Collect results in order
            for future in as_completed(future_to_idx):
                run_idx, grid_and_bar = future.result()
                results[run_idx] = grid_and_bar  # Place results in correct position

        # Flatten the results list
        return [item for sublist in results for item in sublist]
