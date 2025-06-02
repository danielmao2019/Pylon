from typing import List, Dict
import dash
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import numpy as np

from runners.eval_viewer.backend.data_loader import get_common_metrics
from runners.eval_viewer.backend.visualization import create_score_map_figure, create_aggregated_heatmap


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
        if metric is None:
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
