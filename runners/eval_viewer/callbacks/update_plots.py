from typing import List, Dict
import dash
from dash import Input, Output, State, dcc
from dash.exceptions import PreventUpdate
import numpy as np

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
    # Create outputs for each run's score map
    outputs = [Output(f'score-map-{i}', 'children') for i in range(len(log_dirs))]
    
    @app.callback(
        outputs,
        [Input('epoch-slider', 'value'),
         Input('metric-dropdown', 'value')]
    )
    def update_score_maps(epoch: int, metric: str) -> List[dict]:
        """
        Updates the score maps based on selected epoch and metric.
        
        Args:
            epoch: Selected epoch index
            metric: Selected metric name
            log_dirs: List of paths to log directories
            caches: Dictionary mapping log directory to score maps array
            
        Returns:
            figures: List of Plotly figure dictionaries for each run
        """
        if metric is None:
            raise PreventUpdate
        
        figures = []
        for i, log_dir in enumerate(log_dirs):
            # Get score map from cache
            score_maps = caches[log_dir]
            score_map = score_maps[epoch]  # Shape: (C, H, W)
            
            # Create figure
            run_name = log_dir.split('/')[-1]
            fig = create_score_map_figure(score_map, f"{run_name} - {metric}")
            
            # Add to figures list
            figures.append(dcc.Graph(figure=fig))
        
        return figures
