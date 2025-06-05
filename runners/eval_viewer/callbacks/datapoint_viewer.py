"""Callbacks for datapoint viewing functionality."""
from typing import Dict, List, Any
import dash
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import numpy as np

from runners.eval_viewer.backend.initialization import LogDirInfo
from data.viewer.managers.registry import DatasetType
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from utils.builders.builder import build_from_config

import logging
logger = logging.getLogger(__name__)

# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    '2d_change_detection': display_2d_datapoint,
    '3d_change_detection': display_3d_datapoint,
    'point_cloud_registration': display_pcr_datapoint
}


def register_datapoint_viewer_callbacks(
    app,
    metric_names: List[str],
    dataset_cfg: Dict[str, Any],
    dataset_type: DatasetType,
    log_dir_infos: Dict[str, LogDirInfo],
) -> None:
    """Register callbacks for datapoint viewer functionality.

    Args:
        app: Dash application instance
        metric_names: List of metric names
        dataset_cfg: Dataset configuration
        dataset_type: Dataset type
        log_dir_infos: Dict of LogDirInfo instances
    """
    dataset = build_from_config(dataset_cfg)

    @app.callback(
        Output('selected-datapoint', 'children'),
        [Input({'type': 'overlaid-grid-button', 'index': dash.ALL}, 'n_clicks')],
        [State('epoch-slider', 'value'),
         State('metric-dropdown', 'value')]
    )
    def update_selected_datapoint(clicks, epoch: int, metric_name: str):
        if not any(clicks) or epoch is None or metric_name is None:
            raise PreventUpdate

        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered_id
        if isinstance(triggered_id, dict) and 'index' in triggered_id:
            row, col = map(int, triggered_id['index'].split('-'))
        else:
            raise PreventUpdate

        metric_idx = metric_names.index(metric_name)
        score_maps = []
        for info in log_dir_infos.values():
            score_map = info.score_map[epoch, metric_idx]
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
        [Output('score-info-container', 'children'),
         Output('datapoint-visualization-container', 'children')],
        [Input({'type': 'overlaid-grid-button', 'index': dash.ALL}, 'n_clicks'),
         Input({'type': 'individual-grid-button', 'index': dash.ALL}, 'n_clicks')],
    )
    def update_datapoint_viewer(overlaid_clicks, individual_clicks):
        """Update the datapoint viewer when a grid button is clicked.

        Args:
            overlaid_clicks: List of click events from overlaid grid buttons
            individual_clicks: List of click events from individual grid buttons

        Returns:
            Tuple containing:
                - score_info: HTML elements showing score information
                - datapoint_viz: HTML elements showing datapoint visualization
        """
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered_id
        if not isinstance(triggered_id, dict) or 'index' not in triggered_id:
            raise PreventUpdate

        # Parse the button index
        index_parts = triggered_id['index'].split('-')

        if triggered_id['type'] == 'overlaid-grid-button':
            # Overlaid button grid click - use the common dataset from dataset_cfg
            row, col = map(int, index_parts)
            run_idx = None
            current_dataset = dataset  # Use the dataset built from dataset_cfg
            collate_fn = None  # No collate function for overlaid view
            # Calculate side length from overlaid clicks
            side_length = int(np.sqrt(len(overlaid_clicks)))  # Assuming square grid
        else:
            # Individual score map button click - use run-specific dataset and collate function
            run_idx, row, col = map(int, index_parts)
            run_info = list(log_dir_infos.values())[run_idx]
            current_dataset = build_from_config(run_info.dataset_cfg)
            if hasattr(run_info, 'dataloader_cfg') and run_info.dataloader_cfg is not None:
                collate_fn = build_from_config(run_info.dataloader_cfg).collate_fn
            else:
                collate_fn = None
            # Calculate side length from individual clicks
            side_length = int(np.sqrt(len(individual_clicks)))  # Assuming square grid

        # Calculate datapoint index from grid position
        datapoint_idx = row * side_length + col

        # Load and process datapoint
        datapoint = current_dataset[datapoint_idx]
        if collate_fn is not None:
            datapoint = collate_fn([datapoint])

        # Create score info display
        score_info = html.Div([
            html.H4(f"Datapoint {datapoint_idx}"),
            html.P(f"Position: Row {row}, Column {col}"),
            html.P(f"Type: {dataset_type}"),
            html.P(f"Source: {'Individual Run' if run_idx is not None else 'Overlaid View'}"),
            # Add more score information as needed
        ])

        # Get the appropriate display function
        display_func = DISPLAY_FUNCTIONS.get(dataset_type)
        assert display_func is not None, f"No display function found for dataset type: {dataset_type}"
        display = display_func(datapoint)

        return score_info, display
