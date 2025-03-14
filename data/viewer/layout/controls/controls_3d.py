"""UI components for 3D visualization controls."""
from dash import dcc, html


def create_3d_controls(visible=False, point_size=2, point_opacity=0.8):
    """
    Create 3D visualization controls.
    
    Args:
        visible: Whether the controls should be visible
        point_size: Initial point size
        point_opacity: Initial point opacity
        
    Returns:
        html.Div containing 3D controls
    """
    style = {'display': 'block' if visible else 'none', 'margin-top': '20px'}
    
    return html.Div([
        html.H3("3D View Controls", style={'margin-top': '0'}),

        html.Label("Point Size"),
        dcc.Slider(
            id='point-size-slider',
            min=1,
            max=10,
            value=point_size,
            marks={i: str(i) for i in [1, 3, 5, 7, 10]},
            step=0.5
        ),

        html.Label("Point Opacity", style={'margin-top': '20px'}),
        dcc.Slider(
            id='point-opacity-slider',
            min=0.1,
            max=1.0,
            value=point_opacity,
            marks={i/10: str(i/10) for i in range(1, 11, 2)},
            step=0.1
        ),
    ], id='view-controls', style=style)
