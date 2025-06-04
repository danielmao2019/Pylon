"""Layout components for datapoint viewing."""
from dash import html


def create_datapoint_viewer_layout() -> html.Div:
    """Create the layout for the datapoint viewer section.
    
    Returns:
        html.Div: Container for datapoint visualization
    """
    return html.Div([
        html.H3("Datapoint Viewer"),
        html.Div(id="datapoint-viewer-container", children=[
            html.Div([
                # Left side: Score information
                html.Div(id="score-info-container", className="score-info"),
                
                # Right side: Datapoint visualization
                html.Div(id="datapoint-visualization-container", className="datapoint-visualization")
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'gap': '20px',
                'padding': '20px'
            })
        ])
    ], style={
        'border': '1px solid #ddd',
        'borderRadius': '5px',
        'margin': '20px',
        'padding': '20px'
    })
