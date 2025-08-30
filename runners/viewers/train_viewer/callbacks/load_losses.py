from typing import Any, Tuple, List
import os
import dash
from dash import Input, Output, callback, html, dcc
from dash.exceptions import PreventUpdate

from runners.viewers.train_viewer.backend.read_losses import read_losses
from runners.viewers.train_viewer.backend.visualize_losses import visualize_losses


def register_callbacks(app: dash.Dash) -> None:
    """Register callbacks for the training losses viewer.
    
    Args:
        app: Dash application instance
    """
    assert app is not None, "app must not be None"
    assert isinstance(app, dash.Dash), f"app must be Dash instance, got {type(app)}"
    
    @app.callback(
        Output('plots-container', 'children'),
        [
            Input('load-button', 'n_clicks')
        ],
        [
            dash.State('log-dirs-input', 'value')
        ]
    )
    def update_losses_plots(n_clicks: int, log_dirs_text: str) -> List[Any]:
        """Update the losses plots when load button is clicked.
        
        Args:
            n_clicks: Number of times load button was clicked
            log_dirs_text: Comma-separated paths to log directories
            
        Returns:
            List of plot components
        """
        if n_clicks == 0:
            raise PreventUpdate
            
        if not log_dirs_text:
            return []
            
        # Parse comma-separated paths
        log_dirs = [path.strip() for path in log_dirs_text.split(',') if path.strip()]
        
        if not log_dirs:
            return []
            
        plots = []
        
        for log_dir in log_dirs:
            assert os.path.exists(log_dir)
            
            # Load losses (automatically detects available epochs)
            losses = read_losses(log_dir)
            
            # Create visualization with log dir as title
            fig = visualize_losses(losses, title=log_dir)
            
            # Add plot to container
            plots.append(
                dcc.Graph(
                    figure=fig,
                    style={'height': '600px', 'marginBottom': '20px'}
                )
            )
            
        return plots
