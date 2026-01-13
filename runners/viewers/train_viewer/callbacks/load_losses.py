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
        [Output('plots-container', 'children'),
         Output('smoothing-info', 'children')],
        [Input('refresh-button', 'n_clicks'),
         Input('smoothing-slider', 'value')]
    )
    def update_losses_plots(n_clicks: int, smoothing_window: int) -> Tuple[List[Any], str]:
        """Update the losses plots when refresh button is clicked or smoothing changes.

        Args:
            n_clicks: Number of times refresh button was clicked
            smoothing_window: Window size for smoothing

        Returns:
            Tuple of (plot components, smoothing info text)
        """
        # Read log directories from file
        log_dirs_file = os.path.join(os.path.dirname(__file__), '..', 'log_dirs.txt')

        if not os.path.isfile(log_dirs_file):
            return [html.P(f"Log directories file not found: {log_dirs_file}", style={'color': 'red', 'textAlign': 'center'})], ""

        with open(log_dirs_file, 'r') as f:
            log_dirs = [line.strip() for line in f.readlines() if line.strip()]

        if not log_dirs:
            return [html.P("No log directories found in log_dirs.txt", style={'color': 'orange', 'textAlign': 'center'})], ""

        plots = []

        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                plots.append(html.P(f"Log directory does not exist: {log_dir}", style={'color': 'red', 'margin': '10px'}))
                continue

            try:
                # Load losses (automatically detects available epochs)
                losses = read_losses(log_dir)

                # Create visualization with smoothing
                fig = visualize_losses(losses, smoothing_window=smoothing_window, title=log_dir)

                # Add plot to container
                plots.append(
                    dcc.Graph(
                        figure=fig,
                        style={'height': '600px', 'marginBottom': '20px'}
                    )
                )
            except Exception as e:
                plots.append(html.P(f"Error loading {log_dir}: {str(e)}", style={'color': 'red', 'margin': '10px'}))

        # Create smoothing info text
        smoothing_info = f"Window size: {smoothing_window} {'(no smoothing)' if smoothing_window == 1 else 'iterations'}"

        return plots, smoothing_info
