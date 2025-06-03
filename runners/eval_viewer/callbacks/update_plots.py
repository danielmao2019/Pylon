from typing import List, Dict
import numpy as np
import dash
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from runners.eval_viewer.backend.data_loader import get_common_metrics, load_validation_scores
from runners.eval_viewer.backend.visualization import create_score_map_figure, create_aggregated_scores_plot, create_overlaid_score_map


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

def register_callbacks(app: dash.Dash, log_dirs: List[str], caches: Dict[str, np.ndarray]):
    """
    Registers all callbacks for the app.
    """
    # 1. Individual score maps
    outputs = [Output(f'score-map-{i}', 'children') for i in range(len(log_dirs))]
    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_score_maps(epoch: int, metric: str):
        if metric is None or epoch is None:
            raise PreventUpdate
        metrics = sorted(list(get_common_metrics(log_dirs)))
        metric_idx = metrics.index(metric)
        figures = []
        for i, log_dir in enumerate(log_dirs):
            score_maps_cache = caches[log_dir]
            score_map = score_maps_cache[epoch, metric_idx]
            run_name = log_dir.split('/')[-1]
            fig = create_score_map_figure(score_map, f"{run_name} - {metric}")
            figures.append(dcc.Graph(figure=fig))
        return figures

    # 2. Overlaid button grid (overlaid heatmap as buttons)
    @app.callback(
        Output('button-grid-container', 'children'),
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_overlaid_score_map(epoch: int, metric: str):
        if metric is None or epoch is None:
            raise PreventUpdate
        metrics = sorted(list(get_common_metrics(log_dirs)))
        metric_idx = metrics.index(metric)
        score_maps = []
        for log_dir in log_dirs:
            score_maps_cache = caches[log_dir]
            score_map = score_maps_cache[epoch, metric_idx]
            score_maps.append(score_map)
        if score_maps:
            normalized = create_overlaid_score_map(score_maps, f"Common Failure Cases - {metric}")
            side_length = normalized.shape[0]
            buttons = []
            for row in range(side_length):
                for col in range(side_length):
                    value = normalized[row, col]
                    color = get_color_for_score(value, 0.0, 1.0)  # normalized is already in [0, 1]
                    button = html.Button(
                        '',
                        id={'type': 'grid-button', 'index': f'{row}-{col}'},
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
            button_grid = html.Div(buttons, style={
                'display': 'grid',
                'gridTemplateColumns': f'repeat({side_length}, 20px)',
                'gap': '1px',
                'width': 'fit-content',
                'margin': '0 auto'
            })
            return button_grid
        else:
            return html.Div("No data available")

    # 3. Selected datapoint info
    @app.callback(
        Output('selected-datapoint', 'children'),
        [Input({'type': 'grid-button', 'index': dash.ALL}, 'n_clicks')],
        [State('epoch-slider', 'value'),
         State('metric-dropdown', 'value')]
    )
    def update_selected_datapoint(clicks, epoch: int, metric: str):
        if not any(clicks) or epoch is None or metric is None:
            raise PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        row, col = map(int, button_id.split('-'))
        metrics = sorted(list(get_common_metrics(log_dirs)))
        metric_idx = metrics.index(metric)
        score_maps = []
        for log_dir in log_dirs:
            score_maps_cache = caches[log_dir]
            score_map = score_maps_cache[epoch, metric_idx]
            score_maps.append(score_map)
        side_length = score_maps[0].shape[0]
        datapoint_idx = row * side_length + col
        scores = [score_map[row, col] for score_map in score_maps if not np.isnan(score_map[row, col])]
        if scores:
            return html.Div([
                html.H4(f"Datapoint {datapoint_idx}"),
                html.P(f"Position: Row {row}, Column {col}"),
                html.P(f"Number of runs with data: {len(scores)}"),
                html.P(f"Average score: {np.mean(scores):.3f}"),
                html.P(f"Min score: {np.min(scores):.3f}"),
                html.P(f"Max score: {np.max(scores):.3f}"),
            ])
        else:
            return html.Div([
                html.H4(f"Datapoint {datapoint_idx}"),
                html.P("No data available for this position")
            ])

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
