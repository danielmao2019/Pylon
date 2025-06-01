from typing import List
import dash
from dash import html, dcc


def create_controls(max_epoch: int, metrics: List[str]) -> html.Div:
    """
    Creates the control panel with epoch slider and metric dropdown.
    
    Args:
        max_epoch: Maximum epoch index
        metrics: List of available metrics
        
    Returns:
        controls: HTML div containing the controls
    """
    pass


def create_score_maps_grid(num_runs: int) -> html.Div:
    """
    Creates the grid layout for displaying score maps.
    
    Args:
        num_runs: Number of runs to display
        
    Returns:
        grid: HTML div containing the score maps grid
    """
    pass


def create_layout(max_epoch: int, metrics: List[str], num_runs: int) -> html.Div:
    """
    Creates the main dashboard layout.
    
    Args:
        max_epoch: Maximum epoch index
        metrics: List of available metrics
        num_runs: Number of runs to display
        
    Returns:
        layout: HTML div containing the complete layout
    """
    pass
