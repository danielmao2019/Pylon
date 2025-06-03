from typing import List, Dict
import dash
from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate
import numpy as np

from runners.eval_viewer.backend.data_loader import get_common_metrics, load_validation_scores
from runners.eval_viewer.backend.visualization import create_score_map_figure, create_aggregated_heatmap, create_aggregated_scores_plot


def register_callbacks(app: dash.Dash, log_dirs: List[str], caches: Dict[str, np.ndarray]):
    """
    Registers all callbacks for the app.

    Args:
        app: Dash application instance
        log_dirs: List of paths to log directories
        caches: Dictionary mapping log directory to score maps array
    """
    # Create outputs for each run's score map and the aggregated heatmap
    outputs = [Output(f'score-map-{i}', 'children') for i in range(len(log_dirs))]
    outputs.append(Output('aggregated-heatmap', 'children'))

    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_score_maps(epoch: int, metric: str) -> List[dict]:
        """
        Updates the score maps based on selected epoch and metric.

        Args:
            epoch: Selected epoch index
            metric: Selected metric name

        Returns:
            figures: List of Plotly figure dictionaries for each run and the aggregated heatmap
        """
        if metric is None or epoch is None:
            raise PreventUpdate

        # Get sorted list of metrics to ensure consistent indexing
        metrics = sorted(list(get_common_metrics(log_dirs)))
        metric_idx = metrics.index(metric)

        figures = []
        score_maps = []

        # Create individual score maps
        for i, log_dir in enumerate(log_dirs):
            # Get score map from cache
            score_maps_cache = caches[log_dir]  # Shape: (N, C, H, W)

            # Get score map for current epoch and metric
            score_map = score_maps_cache[epoch, metric_idx]  # Shape: (H, W)
            score_maps.append(score_map)

            # Create figure
            run_name = log_dir.split('/')[-1]
            fig = create_score_map_figure(score_map, f"{run_name} - {metric}")
            figures.append(dcc.Graph(figure=fig))

        # Create aggregated heatmap
        agg_fig = create_aggregated_heatmap(score_maps, f"Common Failure Cases - {metric}")
        figures.append(dcc.Graph(figure=agg_fig))

        return figures

    @app.callback(
        Output('aggregated-scores-plot', 'children'),
        [Input('metric-dropdown', 'value')]
    )
    def update_aggregated_scores_plot(metric: str) -> dcc.Graph:
        """
        Updates the aggregated scores plot based on selected metric.

        Args:
            metric: Selected metric name

        Returns:
            figure: Plotly figure dictionary for the aggregated scores plot
        """
        if metric is None:
            raise PreventUpdate

        # Load scores for all epochs from all runs
        epoch_scores = []
        for log_dir in log_dirs:
            run_scores = []
            epoch = 0
            while True:
                try:
                    scores = load_validation_scores(log_dir, epoch)
                    run_scores.append(scores)
                    epoch += 1
                except AssertionError:
                    break
            epoch_scores.append(run_scores)

        # Create figure
        fig = create_aggregated_scores_plot(epoch_scores, log_dirs, metric)
        return dcc.Graph(figure=fig)
