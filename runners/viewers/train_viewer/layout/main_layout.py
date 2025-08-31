from dash import html, dcc


def create_layout() -> html.Div:
    """Create the main layout for the training losses viewer.

    Returns:
        Layout HTML Div with two-column design
    """
    return html.Div([
        html.H1("Training Losses Viewer", style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            # Left Column - Controls
            html.Div([
                html.H3("Controls", style={'marginBottom': '20px'}),

                html.Button('Refresh', id='refresh-button', n_clicks=0,
                           style={'width': '100%', 'marginBottom': '20px', 'padding': '10px'}),

                html.Hr(),

                html.Label("Loss Smoothing:", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
                dcc.Slider(
                    id='smoothing-slider',
                    min=1,
                    max=50,
                    step=1,
                    value=1,
                    marks={1: '1', 10: '10', 25: '25', 50: '50'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(id='smoothing-info', style={'marginTop': '10px', 'fontSize': '12px', 'color': 'gray'})

            ], style={
                'width': '25%',
                'float': 'left',
                'padding': '20px',
                'height': '100vh',
                'overflowY': 'auto',
                'borderRight': '1px solid #ddd',
                'boxSizing': 'border-box'
            }),

            # Right Column - Plots
            html.Div([
                html.Div(id='plots-container')
            ], style={
                'width': '75%',
                'float': 'right',
                'height': '100vh',
                'overflowY': 'auto',
                'padding': '20px',
                'boxSizing': 'border-box'
            })

        ], style={'display': 'flex', 'height': '100vh'})

    ])
