"""Callbacks for datapoint viewing functionality."""
from typing import Dict, List, Any
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash
from data.viewer.backend.backend import DatasetType
from runners.viewers.eval_viewer.backend.initialization import LogDirInfo, load_debug_outputs
from utils.builders.builder import build_from_config

import logging
logger = logging.getLogger(__name__)


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
    # Input validation following CLAUDE.md fail-fast patterns
    assert app is not None, "app must not be None"
    assert isinstance(app, dash.Dash), f"app must be dash.Dash instance, got {type(app)}"

    assert dataset_cfg is not None, "dataset_cfg must not be None"
    assert isinstance(dataset_cfg, dict), f"dataset_cfg must be dict, got {type(dataset_cfg)}"
    assert len(dataset_cfg) > 0, f"dataset_cfg must not be empty"

    assert dataset_type is not None, "dataset_type must not be None"
    assert isinstance(dataset_type, str), f"dataset_type must be str, got {type(dataset_type)}"

    assert log_dir_infos is not None, "log_dir_infos must not be None"
    assert isinstance(log_dir_infos, dict), f"log_dir_infos must be dict, got {type(log_dir_infos)}"
    assert len(log_dir_infos) > 0, f"log_dir_infos must not be empty"
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
        # PRESERVE ORIGINAL PreventUpdate logic - DO NOT change this condition
        if not any(overlaid_clicks) and not any(individual_clicks) or epoch is None or metric_name is None:
            raise PreventUpdate

        # Input validation following CLAUDE.md fail-fast patterns - AFTER PreventUpdate
        assert overlaid_clicks is not None, "overlaid_clicks must not be None"
        assert isinstance(overlaid_clicks, list), f"overlaid_clicks must be list, got {type(overlaid_clicks)}"

        assert individual_clicks is not None, "individual_clicks must not be None"
        assert isinstance(individual_clicks, list), f"individual_clicks must be list, got {type(individual_clicks)}"

        assert epoch is not None, "epoch must not be None"
        assert isinstance(epoch, int), f"epoch must be int, got {type(epoch)}"
        assert epoch >= 0, f"epoch must be non-negative, got {epoch}"

        assert metric_name is not None, "metric_name must not be None"
        assert isinstance(metric_name, str), f"metric_name must be str, got {type(metric_name)}"

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
            dataloader = build_from_config(run_info.dataloader_cfg, dataset=current_dataset)
            collate_fn = dataloader.collate_fn
            datapoint = current_dataset[datapoint_idx]
            datapoint = collate_fn([datapoint])  # Apply collate function to single datapoint

            # Load debug outputs if available
            log_dir = list(log_dir_infos.keys())[run_idx]
            if run_info.runner_type == 'trainer':
                # For trainer: load from epoch directory
                epoch_dir = f"{log_dir}/epoch_{epoch}"
                debug_outputs = load_debug_outputs(epoch_dir)
            else:
                # For evaluator: load from root directory
                debug_outputs = load_debug_outputs(log_dir)

            # debug_outputs is a dict mapping datapoint_idx to debug_data, or None
            if debug_outputs and datapoint_idx in debug_outputs:
                datapoint['debug'] = debug_outputs[datapoint_idx]

        # Use current dataset instance display method directly
        assert hasattr(current_dataset, 'display_datapoint'), f"Dataset {type(current_dataset).__name__} must have display_datapoint method"
        display = current_dataset.display_datapoint(datapoint)

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
