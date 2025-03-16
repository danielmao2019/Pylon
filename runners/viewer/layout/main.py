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
            # Quick Navigation Sliders
            html.Div([
                # Epoch slider
                html.Div([
                    html.Div("Epoch Navigation", 
                            style={'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                  'color': '#2c3e50', 'textAlign': 'center'}),
                    dcc.Slider(
                        id='epoch-slider',
                        min=0,
                        max=100,  # Will be updated dynamically
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(0, 101, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'marginBottom': '20px'}),
                
                # Iteration slider
                html.Div([
                    html.Div("Iteration in Current Epoch", 
                            style={'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '10px',
                                  'color': '#2c3e50', 'textAlign': 'center'}),
                    dcc.Slider(
                        id='iteration-slider',
                        min=0,
                        max=100,  # Will be updated dynamically
                        step=1,
                        value=0,
                        marks={i: str(i) for i in range(0, 101, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ]),
            ], style={'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginBottom': '20px'}),

            # Navigation controls with better styling
            html.Div([
                # Epoch navigation
                html.Div([
                    html.Button("◀ Previous Epoch", 
                              id="btn-prev-epoch",
                              style={'marginRight': '10px', 'backgroundColor': '#9b59b6', 'color': 'white', 
                                    'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                    html.Button("Next Epoch ▶", 
                              id="btn-next-epoch",
                              style={'backgroundColor': '#9b59b6', 'color': 'white', 
                                    'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                ], style={'marginBottom': '10px', 'textAlign': 'center'}),
                
                # Iteration navigation
                html.Div([
                    html.Button("◀ Previous Iteration", 
                              id="btn-prev-iter",
                              style={'marginRight': '10px', 'backgroundColor': '#3498db', 'color': 'white', 
                                    'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                    html.Button("Next Iteration ▶", 
                              id="btn-next-iter",
                              style={'backgroundColor': '#3498db', 'color': 'white', 
                                    'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                ], style={'marginBottom': '10px', 'textAlign': 'center'}),
                
                # Sample navigation
                html.Div([
                    html.Button("◀ Previous Sample", 
                              id="btn-prev-sample",
                              style={'marginRight': '10px', 'backgroundColor': '#2ecc71', 'color': 'white', 
                                    'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                    html.Button("Next Sample ▶", 
                              id="btn-next-sample",
                              style={'backgroundColor': '#2ecc71', 'color': 'white', 
                                    'border': 'none', 'padding': '10px 15px', 'borderRadius': '4px'}),
                ], style={'textAlign': 'center'}),
            ], style={'marginBottom': '20px'}),
        ]),
        
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
        
        # Image displays with two-column layout
        html.Div([
            # Left column - Input Images
            html.Div([
                html.H2("Input Images", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                # Input images container
                html.Div([
                    # Image 1
                    html.Div([
                        html.H3("Image 1", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id="input-image-1", style={'height': '400px'})
                    ], style={'marginBottom': '20px'}),
                    
                    # Image 2
                    html.Div([
                        html.H3("Image 2", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id="input-image-2", style={'height': '400px'})
                    ])
                ])
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'marginRight': '2%'}),
            
            # Right column - Change Maps
            html.Div([
                html.H2("Change Maps", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '20px'}),
                # Change maps container
                html.Div([
                    # Predicted Changes
                    html.Div([
                        html.H3("Predicted Changes", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id="pred-change-map", style={'height': '400px'})
                    ], style={'marginBottom': '20px'}),
                    
                    # Ground Truth
                    html.Div([
                        html.H3("Ground Truth", style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '10px'}),
                        dcc.Graph(id="gt-change-map", style={'height': '400px'})
                    ])
                ])
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top',
                     'backgroundColor': '#ffffff', 'padding': '20px', 'borderRadius': '8px',
                     'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'})
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9', 'minHeight': '100vh'})
