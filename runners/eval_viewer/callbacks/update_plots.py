from typing import List, Dict
import dash
from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate
import numpy as np

from runners.eval_viewer.backend.data_loader import get_common_metrics, load_validation_scores
from runners.eval_viewer.backend.visualization import create_score_map_figure, create_aggregated_heatmap, create_aggregated_scores_plot

from utils.data_loader import load_validation_scores, extract_metric_scores
from utils.visualization import create_score_map, create_score_map_figure


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
    outputs.append(Output('aggregated-heatmap-graph', 'children'))
    outputs.append(Output('selected-datapoint', 'children'))

    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value'),
         Input('aggregated-heatmap-graph', 'clickData')],
        [State('selected-datapoint', 'children')]
    )
    def update_score_maps(epoch: int, metric: str, click_data: dict, prev_selection: dict) -> List[dict]:
        """
        Updates the score maps based on selected epoch and metric.

        Args:
            epoch: Selected epoch index
            metric: Selected metric name
            click_data: Data from heatmap click event
            prev_selection: Previous datapoint selection

        Returns:
            figures: List of Plotly figure dictionaries for each run and the aggregated heatmap,
                    plus selected datapoint information
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
        figures.append(dcc.Graph(
            figure=agg_fig,
            id='aggregated-heatmap-graph',
            config={'displayModeBar': True},
            style={'height': '600px'}
        ))

        # Handle selected datapoint
        if click_data is not None and 'points' in click_data and len(click_data['points']) > 0:
            # Get clicked point coordinates
            point = click_data['points'][0]
            row, col = point['y'], point['x']
            
            # Calculate datapoint index
            side_length = score_maps[0].shape[0]
            datapoint_idx = row * side_length + col
            
            # Get scores for this datapoint across all runs
            scores = []
            for score_map in score_maps:
                if not np.isnan(score_map[row, col]):
                    scores.append(score_map[row, col])
            
            if scores:
                # Create datapoint info display
                datapoint_info = html.Div([
                    html.H4(f"Datapoint {datapoint_idx}"),
                    html.P(f"Position: Row {row}, Column {col}"),
                    html.P(f"Number of runs with data: {len(scores)}"),
                    html.P(f"Average score: {np.mean(scores):.3f}"),
                    html.P(f"Min score: {np.min(scores):.3f}"),
                    html.P(f"Max score: {np.max(scores):.3f}"),
                ])
            else:
                datapoint_info = html.Div([
                    html.H4(f"Datapoint {datapoint_idx}"),
                    html.P("No data available for this position")
                ])
        elif prev_selection is not None:
            # Keep previous selection if no new click
            datapoint_info = prev_selection
        else:
            datapoint_info = html.Div([
                html.P("Click on a cell in the heatmap to view datapoint details")
            ])

        figures.append(datapoint_info)
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
