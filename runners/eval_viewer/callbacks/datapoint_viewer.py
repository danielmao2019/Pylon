"""Callbacks for datapoint viewing functionality."""
from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate
import numpy as np
import logging
from typing import Dict, List, Optional, Union, Literal

from runners.eval_viewer.backend.datapoint_viewer import DatapointViewer
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint
from data.viewer.layout.display.display_pcr import display_pcr_datapoint
from data.viewer.managers.registry import get_dataset_type, DatasetType

logger = logging.getLogger(__name__)

# Mapping of dataset types to their display functions
DISPLAY_FUNCTIONS = {
    '2d_change_detection': display_2d_datapoint,
    '3d_change_detection': display_3d_datapoint,
    'point_cloud_registration': display_pcr_datapoint
}

def register_datapoint_viewer_callbacks(app, datapoint_viewer: DatapointViewer):
    """Register callbacks for datapoint viewer functionality.
    
    Args:
        app: Dash application instance
        datapoint_viewer: DatapointViewer instance
    """
    @app.callback(
        [Output('score-info-container', 'children'),
         Output('datapoint-visualization-container', 'children')],
        [Input({'type': 'grid-button', 'index': dash.ALL}, 'n_clicks')],
        [State('epoch-slider', 'value'),
         State('metric-dropdown', 'value'),
         State('log-dirs', 'value')]
    )
    def update_datapoint_viewer(clicks, epoch: int, metric: str, log_dir: str):
        """Update the datapoint viewer when a grid button is clicked.
        
        Args:
            clicks: List of click events from grid buttons
            epoch: Current epoch number
            metric: Selected metric name
            log_dir: Path to the log directory being viewed
            
        Returns:
            Tuple containing:
                - score_info: HTML elements showing score information
                - datapoint_viz: HTML elements showing datapoint visualization
        """
        if not any(clicks) or epoch is None or metric is None or log_dir is None:
            raise PreventUpdate
            
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        triggered_id = ctx.triggered_id
        if not isinstance(triggered_id, dict) or 'index' not in triggered_id:
            raise PreventUpdate
            
        # Get row and column from button index
        row, col = map(int, triggered_id['index'].split('-'))
        
        try:
            # Calculate datapoint index from grid position
            side_length = int(np.sqrt(len(clicks)))  # Assuming square grid
            datapoint_idx = row * side_length + col
            
            # Load datapoint
            datapoint = datapoint_viewer.load_datapoint(log_dir, datapoint_idx)
            
            # Get dataset type using registry function
            try:
                dataset_type = get_dataset_type(datapoint_viewer.current_dataset)
            except ValueError as e:
                logger.error(f"Error determining dataset type: {str(e)}")
                return html.Div(f"Error: {str(e)}"), html.Div()
            
            # Create score info display
            score_info = html.Div([
                html.H4(f"Datapoint {datapoint_idx}"),
                html.P(f"Position: Row {row}, Column {col}"),
                html.P(f"Dataset: {datapoint_viewer.current_dataset}"),
                html.P(f"Type: {dataset_type}"),
                # Add more score information as needed
            ])
            
            # Get the appropriate display function
            display_func = DISPLAY_FUNCTIONS.get(dataset_type)
            if display_func is None:
                logger.error(f"No display function found for dataset type: {dataset_type}")
                return score_info, html.Div(f"Error: Unsupported dataset type: {dataset_type}")
            
            # Create datapoint visualization
            try:
                if dataset_type == 'point_cloud_registration':
                    display = display_func(datapoint, point_size=2.0, point_opacity=0.8, camera_state={}, radius=1.0)
                elif dataset_type == '3d_change_detection':
                    display = display_func(datapoint, point_size=2.0, point_opacity=0.8, class_labels={}, camera_state={})
                else:  # 2d_change_detection
                    display = display_func(datapoint)
                
                return score_info, display
                
            except Exception as e:
                logger.error(f"Error creating display: {str(e)}")
                return score_info, html.Div(f"Error creating visualization: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error loading datapoint: {str(e)}")
            return html.Div(f"Error loading datapoint: {str(e)}"), html.Div()
