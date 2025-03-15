"""Navigation-related callbacks for the runners viewer."""
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import torch
from runners.viewer.utils import tensor_to_image, create_image_figure

logger = logging.getLogger(__name__)


def register_navigation_callbacks(app, state):
    """Register navigation callbacks for the viewer application.
    
    Args:
        app: Dash application instance
        state: State management object
    """
    @app.callback(
        [Output("iteration-display", "children"),
         Output("sample-display", "children"),
         Output("input-image-1", "figure"),
         Output("input-image-2", "figure"),
         Output("pred-change-map", "figure"),
         Output("gt-change-map", "figure")],
        [Input("btn-prev-iter", "n_clicks"),
         Input("btn-next-iter", "n_clicks"),
         Input("btn-prev-sample", "n_clicks"),
         Input("btn-next-sample", "n_clicks")]
    )
    def update_display(prev_iter_clicks, next_iter_clicks, prev_sample_clicks, next_sample_clicks):
        """Update the display based on navigation button clicks.
        
        Returns:
            Tuple containing:
                - Iteration display text
                - Sample display text
                - Input image 1 figure
                - Input image 2 figure
                - Predicted change map figure
                - Ground truth change map figure
        """
        # Check if callback was triggered
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        # Get button that triggered the callback
        button_id = ctx.triggered_id
        
        # Update state based on button click
        try:
            if button_id == "btn-next-iter":
                state.next_iteration()
            elif button_id == "btn-prev-iter":
                state.prev_iteration()
            elif button_id == "btn-next-sample":
                state.next_sample()
            elif button_id == "btn-prev-sample":
                state.prev_sample()
        except Exception as e:
            logger.error(f"Error updating state: {str(e)}")
            raise PreventUpdate
        
        try:
            # Get current data
            data = state.get_current_data()
            nav_info = state.get_navigation_info()
            
            # Convert input images to displayable format
            input1_array = tensor_to_image(data["input1"])
            input2_array = tensor_to_image(data["input2"])
            
            # Convert predictions and ground truth to RGB
            pred_rgb = state.class_to_rgb(data["pred"])
            gt_rgb = state.class_to_rgb(data["gt"])
            pred_array = tensor_to_image(pred_rgb / 255.0)
            gt_array = tensor_to_image(gt_rgb / 255.0)
            
            # Create figures
            input1_fig = create_image_figure(input1_array, title="Input Image 1")
            input2_fig = create_image_figure(input2_array, title="Input Image 2")
            pred_fig = create_image_figure(pred_array, title="Predicted Changes")
            gt_fig = create_image_figure(gt_array, title="Ground Truth")
            
            # Create display text
            iter_text = f"Training Iteration: {nav_info['current_iteration']} / {nav_info['total_iterations']}"
            sample_text = f"Sample: {nav_info['current_sample']} / {nav_info['batch_size']}"
            
            return (
                iter_text,
                sample_text,
                input1_fig,
                input2_fig,
                pred_fig,
                gt_fig
            )
            
        except Exception as e:
            logger.error(f"Error updating display: {str(e)}")
            # Return empty figures on error
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                margin=dict(l=0, r=0, t=0, b=0)
            )
            return (
                "Error: Failed to update display",
                "Error: Failed to update display",
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig
            )
