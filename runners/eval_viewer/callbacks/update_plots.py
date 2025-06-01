from typing import List
import dash
from dash import Input, Output, State
from dash.exceptions import PreventUpdate


def register_callbacks(app: dash.Dash, log_dirs: List[str]):
    """
    Registers all callbacks for the app.
    
    Args:
        app: Dash application instance
        log_dirs: List of paths to log directories
    """
    pass


def update_score_maps(
    epoch: int,
    metric: str,
    log_dirs: List[str]
) -> List[dict]:
    """
    Updates the score maps based on selected epoch and metric.
    
    Args:
        epoch: Selected epoch index
        metric: Selected metric name
        log_dirs: List of paths to log directories
        
    Returns:
        figures: List of Plotly figure dictionaries for each run
    """
    pass
