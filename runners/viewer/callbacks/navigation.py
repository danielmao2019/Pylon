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
        [Output('epoch-slider', 'max'),
         Output('iteration-slider', 'max')],
        [Input('iteration-display', 'children')]
    )
    def update_sliders_max(iteration_display):
        """Update both sliders' maximum values."""
        try:
            total_epochs = state.total_epochs
            iterations_per_epoch = len(state.train_dataloader)
            return total_epochs - 1, iterations_per_epoch - 1
        except Exception as e:
            logger.error(f"Error updating slider max: {str(e)}")
            return 100, 100  # Default values
            
    @app.callback(
        [Output("epoch-display", "children"),
         Output("iteration-display", "children"),
         Output("sample-display", "children"),
         Output("input-image-1", "figure"),
         Output("input-image-2", "figure"),
         Output("pred-change-map", "figure"),
         Output("gt-change-map", "figure"),
         Output("epoch-slider", "value"),
         Output("iteration-slider", "value")],
        [Input("btn-prev-iter", "n_clicks"),
         Input("btn-next-iter", "n_clicks"),
         Input("btn-prev-sample", "n_clicks"),
         Input("btn-next-sample", "n_clicks"),
         Input("epoch-slider", "value"),
         Input("iteration-slider", "value")]
    )
    def update_display(prev_iter_clicks, next_iter_clicks, prev_sample_clicks, next_sample_clicks, epoch_value, iter_value):
        """Update the display based on navigation controls or slider values."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        try:
            # Handle slider navigation
            if trigger_id in ["epoch-slider", "iteration-slider"]:
                current_epoch = state.current_epoch
                current_iter = state.current_iteration
                
                if trigger_id == "epoch-slider" and epoch_value > current_epoch:
                    # Move forward to desired epoch
                    for _ in range(epoch_value - current_epoch):
                        state.next_epoch()
                elif trigger_id == "iteration-slider" and iter_value > current_iter:
                    # Move forward to desired iteration in current epoch
                    for _ in range(iter_value - current_iter):
                        state.next_iteration()
            # Handle button navigation
            else:
                if trigger_id == "btn-next-iter":
                    state.next_iteration()
                elif trigger_id == "btn-prev-iter":
                    state.prev_iteration()
                elif trigger_id == "btn-next-sample":
                    state.next_sample()
                elif trigger_id == "btn-prev-sample":
                    state.prev_sample()
                    
            # Get current data and info
            data = state.get_current_data()
            nav_info = state.get_navigation_info()
            
            # Convert input images to displayable format
            input1_array = tensor_to_image(data["input1"])
            input2_array = tensor_to_image(data["input2"])
            
            # Convert predictions and ground truth to RGB
            pred_array = state.class_to_rgb(data["pred"]) / 255.0
            gt_array = state.class_to_rgb(data["gt"]) / 255.0
            
            # Create figures
            input1_fig = create_image_figure(input1_array, title="Input Image 1")
            input2_fig = create_image_figure(input2_array, title="Input Image 2")
            pred_fig = create_image_figure(pred_array, title="Predicted Changes")
            gt_fig = create_image_figure(gt_array, title="Ground Truth")
            
            # Create display text
            epoch_text = f"Epoch: {nav_info['current_epoch']}"
            iter_text = f"Training Iteration: {nav_info['current_iteration']} / {nav_info['total_iterations']}"
            sample_text = f"Sample: {nav_info['current_sample']} / {nav_info['batch_size']}"
            
            return (
                epoch_text,
                iter_text,
                sample_text,
                input1_fig,
                input2_fig,
                pred_fig,
                gt_fig,
                nav_info['current_epoch'],  # Update epoch slider
                nav_info['current_iteration']  # Update iteration slider
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
                "Error: Failed to update display",
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                0,  # Reset epoch slider
                0   # Reset iteration slider
            )
