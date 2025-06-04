"""Layout components for datapoint viewing."""
from dash import html


def create_datapoint_viewer_layout() -> html.Div:
    """Create the layout for the datapoint viewer section.

    Returns:
        html.Div: Container for datapoint visualization
    """
    return html.Div([
        html.H3("Datapoint Viewer", style={'marginTop': '0'}),
        html.Div(id="datapoint-viewer-container", children=[
            html.Div([
                # Left side: Score information
                html.Div(id="score-info-container", className="score-info", style={'flex': '1'}),

                # Right side: Datapoint visualization
                html.Div(id="datapoint-visualization-container", className="datapoint-visualization", style={'flex': '2'})
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
            })
        ])
    ], style={
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'padding': '20px'
    })
