from typing import Any, List
import dash
from dash import html, dcc


def create_layout() -> html.Div:
    """Create the main layout for the training losses viewer.
    
    Returns:
        Layout HTML Div
    """
    return html.Div([
        html.H1("Training Losses Viewer", style={'textAlign': 'center'}),
        
        html.Div([
            html.Label("Log Directory Paths (comma-separated):"),
            dcc.Textarea(
                id='log-dirs-input',
                placeholder='Enter paths to log directories, separated by commas',
                style={'width': '600px', 'height': '80px', 'marginRight': '10px'}
            ),
            html.Button('Load Losses', id='load-button', n_clicks=0)
        ], style={'margin': '20px', 'textAlign': 'center'}),
        
        
        html.Div(id='plots-container', style={'margin': '20px'})
    ])
