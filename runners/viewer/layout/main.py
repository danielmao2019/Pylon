from dash import html, dcc


def create_main_layout():
    """Create the main layout for the viewer application."""
    return html.Div([
        # Navigation controls
        html.Div([
            html.Div([
                html.Button("Previous Iteration", id="btn-prev-iter"),
                html.Button("Next Iteration", id="btn-next-iter"),
                html.Button("Previous Sample", id="btn-prev-sample"),
                html.Button("Next Sample", id="btn-next-sample"),
            ], style={'marginBottom': '10px'}),
            
            # Status displays
            html.Div([
                html.Div(id="epoch-display", style={'marginRight': '20px', 'display': 'inline-block'}),
                html.Div(id="iteration-display", style={'marginRight': '20px', 'display': 'inline-block'}),
                html.Div(id="sample-display", style={'display': 'inline-block'})
            ])
        ], style={'marginBottom': '20px'}),
        
        # Image displays
        html.Div([
            html.Div([
                dcc.Graph(id="input-image-1")
            ], style={'width': '25%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id="input-image-2")
            ], style={'width': '25%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id="pred-change-map")
            ], style={'width': '25%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id="gt-change-map")
            ], style={'width': '25%', 'display': 'inline-block'})
        ])
    ])
