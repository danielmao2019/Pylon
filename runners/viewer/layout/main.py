from dash import html, dcc


def create_main_layout():
    """Create the main layout of the viewer."""
    return html.Div([
        # Header
        html.H1("Training Viewer", style={'textAlign': 'center'}),
        
        # Navigation buttons
        html.Div([
            html.Button("Previous", id="btn-prev", n_clicks=0),
            html.Button("Next", id="btn-next", n_clicks=0),
            html.Div(id="iteration-display", children="Iteration: 0")
        ], style={'textAlign': 'center', 'margin': '20px'}),
        
        # Image displays
        html.Div([
            # Input images
            html.Div([
                html.H3("Input Image 1"),
                dcc.Graph(id="input-image-1"),
                html.H3("Input Image 2"),
                dcc.Graph(id="input-image-2"),
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            # Prediction and ground truth
            html.Div([
                html.H3("Predicted Change Map"),
                dcc.Graph(id="pred-change-map"),
                html.H3("Ground Truth Change Map"),
                dcc.Graph(id="gt-change-map"),
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ])
