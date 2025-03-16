from dash import html, dcc


def create_main_layout():
    """Create the main layout for the viewer application."""
    return html.Div([
        # Header
        html.Div([
            html.H1("Training Progress Viewer", 
                   style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'})
        ]),

        # Navigation and Status Panel
        html.Div([
            # Navigation controls with better styling
            html.Div([
                html.Button("◀ Previous Iteration", 
                           id="btn-prev-iter",
                           style={'marginRight': '10px', 'backgroundColor': '#3498db', 'color': 'white', 
                                 'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                html.Button("Next Iteration ▶", 
                           id="btn-next-iter",
                           style={'marginRight': '20px', 'backgroundColor': '#3498db', 'color': 'white', 
                                 'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                html.Button("◀ Previous Sample", 
                           id="btn-prev-sample",
                           style={'marginRight': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 
                                 'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                html.Button("Next Sample ▶", 
                           id="btn-next-sample",
                           style={'backgroundColor': '#2ecc71', 'color': 'white', 
                                 'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
            ], style={'marginBottom': '20px', 'textAlign': 'center'}),
            
            # Status displays with better styling
            html.Div([
                html.Div([
                    html.Div("Current Status", 
                            style={'fontSize': '18px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                  'color': '#2c3e50', 'textAlign': 'center'}),
                    html.Div([
                        html.Div([
                            html.Span("Epoch: ", style={'fontWeight': 'bold'}),
                            html.Span(id="epoch-display")
                        ], style={'backgroundColor': '#f1c40f', 'padding': '10px 20px', 'borderRadius': '4px',
                                'marginRight': '20px', 'display': 'inline-block'}),
                        html.Div([
                            html.Span("Iteration: ", style={'fontWeight': 'bold'}),
                            html.Span(id="iteration-display")
                        ], style={'backgroundColor': '#e74c3c', 'padding': '10px 20px', 'borderRadius': '4px',
                                'marginRight': '20px', 'display': 'inline-block'}),
                        html.Div([
                            html.Span("Sample: ", style={'fontWeight': 'bold'}),
                            html.Span(id="sample-display")
                        ], style={'backgroundColor': '#27ae60', 'padding': '10px 20px', 'borderRadius': '4px',
                                'display': 'inline-block'})
                    ], style={'textAlign': 'center'})
                ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '8px',
                         'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
            ], style={'marginBottom': '30px'}),
        ]),
        
        # Image displays with better styling
        html.Div([
            html.Div([
                html.H3("Input Image 1", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id="input-image-1")
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3("Input Image 2", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id="input-image-2")
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3("Predicted Changes", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id="pred-change-map")
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '10px'}),
            
            html.Div([
                html.H3("Ground Truth", style={'textAlign': 'center', 'color': '#2c3e50'}),
                dcc.Graph(id="gt-change-map")
            ], style={'width': '25%', 'display': 'inline-block', 'padding': '10px'})
        ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                  'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'minHeight': '100vh'})
