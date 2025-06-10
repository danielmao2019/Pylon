"""Callbacks for datapoint viewing functionality."""
from typing import Dict, List, Any
import dash
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import numpy as np

from runners.eval_viewer.backend.initialization import LogDirInfo
from data.viewer.managers.registry import DatasetType
from data.viewer.layout.display.display_semseg import display_semseg_datapoint
from data.viewer.layout.display.display_2dcd import display_2dcd_datapoint
from data.viewer.layout.display.display_3dcd import display_3dcd_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from utils.builders.builder import build_from_config

import logging
logger = logging.getLogger(__name__)

# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    'semseg': display_semseg_datapoint,
    '2dcd': display_2dcd_datapoint,
    '3dcd': display_3dcd_datapoint,
    'pcr': display_pcr_datapoint,
}


def register_datapoint_viewer_callbacks(
    app,
    dataset_cfg: Dict[str, Any],
    dataset_type: DatasetType,
    log_dir_infos: Dict[str, LogDirInfo],
) -> None:
    """Register callbacks for datapoint viewer functionality.

    Args:
        app: Dash application instance
        dataset_cfg: Dataset configuration
        dataset_type: Dataset type
        log_dir_infos: Dict of LogDirInfo instances
    """
    dataset = build_from_config(dataset_cfg)

    @app.callback(
        Output('datapoint-display', 'children'),
        [Input({'type': 'overlaid-grid-button', 'index': dash.ALL}, 'n_clicks'),
         Input({'type': 'individual-grid-button', 'index': dash.ALL}, 'n_clicks')],
        [State('epoch-slider', 'value'),
         State('metric-dropdown', 'value')]
    )
    def update_selected_datapoint(overlaid_clicks, individual_clicks, epoch: int, metric_name: str):
        """Update the selected datapoint display when a grid button is clicked.

        Args:
            overlaid_clicks: List of click events from overlaid grid buttons
            individual_clicks: List of click events from individual grid buttons
            epoch: Current epoch
            metric_name: Current metric name

        Returns:
            HTML elements showing datapoint information and visualization
        """
        if not any(overlaid_clicks) and not any(individual_clicks) or epoch is None or metric_name is None:
            raise PreventUpdate

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
            run_idx = None
            datapoint_idx = int(index_parts[0])
            current_dataset = dataset  # Use the dataset built from dataset_cfg
            datapoint = current_dataset[datapoint_idx]
        else:
            # Individual score map button click - use run-specific dataset and dataloader
            run_idx = int(index_parts[0])
            datapoint_idx = int(index_parts[1])
            run_info = list(log_dir_infos.values())[run_idx]
            current_dataset = build_from_config(run_info.dataset_cfg)
            dataloader = build_from_config(run_info.dataloader_cfg)
            collate_fn = dataloader.collate_fn
            datapoint = collate_fn([datapoint])  # Apply collate function to single datapoint

        # Get the appropriate display function
        display_func = DISPLAY_FUNCTIONS.get(dataset_type)
        assert display_func is not None, f"No display function found for dataset type: {dataset_type}"
        display = display_func(datapoint)

        # Create combined display with info and visualization
        return html.Div([
            # Info section
            html.Div([
                html.H4(f"Datapoint {datapoint_idx}"),
                html.P(f"Type: {dataset_type}"),
                html.P(f"Source: {'Individual Run' if run_idx is not None else 'Overlaid View'}"),
            ], style={'marginBottom': '20px'}),
            
            # Visualization section
            html.Div(display)
        ])
