"""UI components for 3D visualization controls."""
from dash import dcc, html
from data.viewer.utils.settings_config import ViewerSettings


def create_3d_controls(visible=False, **kwargs):
    """
    Create 3D visualization controls.

    Args:
        visible: Whether the controls should be visible
        **kwargs: Optional overrides for default settings

    Returns:
        html.Div containing 3D controls
    """
    # Get default settings and apply any overrides
    settings = ViewerSettings.get_3d_settings_with_defaults(kwargs)
    
    style = {'display': 'block' if visible else 'none', 'margin-top': '20px'}

    return html.Div([
        html.H3("3D View Controls", style={'margin-top': '0'}),

        # LOD Controls
        html.Div([
            html.Label("Level of Detail (LOD) Mode", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            dcc.Dropdown(
                id='lod-type-dropdown',
                options=ViewerSettings.LOD_TYPE_OPTIONS,
                value=settings['lod_type'],
                clearable=False,
                style={'margin-bottom': '10px'}
            ),
            html.Div(id='lod-info-display', style={'font-size': '12px', 'color': '#666', 'margin-bottom': '15px'})
        ]),

        # Camera controls
        html.Button(
            'Reset Camera View',
            id='reset-camera-button',
            style={'width': '100%', 'padding': '10px', 'background-color': '#007bff', 'color': 'white', 'border': 'none', 'border-radius': '5px', 'cursor': 'pointer', 'margin-bottom': '20px'}
        ),

        html.Label("Point Size"),
        dcc.Slider(
            id='point-size-slider',
            min=1,
            max=10,
            value=settings['point_size'],
            marks={i: str(i) for i in [1, 3, 5, 7, 10]},
            step=0.5
        ),

        html.Label("Point Opacity", style={'margin-top': '20px'}),
        dcc.Slider(
            id='point-opacity-slider',
            min=0.1,
            max=1.0,
            value=settings['point_opacity'],
            marks={i/10: str(i/10) for i in range(1, 11, 2)},
            step=0.1
        ),

        # PCR-specific controls
        html.Div([
            # Radius slider for symmetric difference computation
            html.Label("Symmetric Difference Radius", style={'margin-top': '20px'}),
            dcc.Slider(
                id='radius-slider',
                min=0.0,
                max=1.0,
                value=settings['sym_diff_radius'],
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                step=0.01
            ),

            # Correspondence radius slider
            html.Label("Correspondence Radius", style={'margin-top': '20px'}),
            dcc.Slider(
                id='correspondence-radius-slider',
                min=0.0,
                max=1.0,
                value=settings['corr_radius'],
                marks={i/10: str(i/10) for i in range(0, 11, 2)},
                step=0.01
            ),
        ], id='pcr-controls', style={'display': 'none'}),  # Hidden by default, shown only for PCR datasets

        # Density controls (only visible when lod_type == 'none')
        html.Div([
            html.Label("Density", style={'margin-top': '20px'}),
            dcc.Slider(
                id='density-slider',
                min=1,
                max=100,
                value=settings['density_percentage'],
                marks={i: f"{i}%" for i in [1, 25, 50, 75, 100]},
                step=1,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], id='density-controls', style={'display': 'none'}),  # Hidden by default, shown only when LOD type is 'none'
    ], id='view-controls', style=style)
