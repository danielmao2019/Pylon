#!/usr/bin/env python3
"""
Layout module for LiDAR simulation cropping visualization.
Defines the user interface components and layout structure.
"""

from dash import dcc, html
from typing import Dict, List, Any

from .backend import LiDARVisualizationBackend


class LiDARVisualizationLayout:
    """Layout class for creating the web app UI components."""
    
    def __init__(self, backend: LiDARVisualizationBackend):
        """Initialize the layout with backend reference.
        
        Args:
            backend: Backend instance for getting available options
        """
        self.backend = backend
        self.options = backend.get_available_options()
    
    def create_header(self) -> html.Div:
        """Create the header section.
        
        Returns:
            Dash HTML component for header
        """
        return html.H1(
            "LiDAR Simulation Cropping - Interactive Visualization",
            style={'textAlign': 'center', 'marginBottom': 30}
        )
    
    def create_controls(self) -> html.Div:
        """Create the control panel with dropdowns.
        
        Returns:
            Dash HTML component for controls
        """
        return html.Div([
            # Point cloud selection
            html.Div([
                html.Label(
                    "Point Cloud:", 
                    style={'fontWeight': 'bold', 'marginBottom': 10}
                ),
                dcc.Dropdown(
                    id='point-cloud-dropdown',
                    options=self.options['point_clouds'],
                    value='cube',
                    style={'width': '100%'},
                    clearable=False
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            # Crop type selection
            html.Div([
                html.Label(
                    "Crop Type:", 
                    style={'fontWeight': 'bold', 'marginBottom': 10}
                ),
                dcc.Dropdown(
                    id='crop-type-dropdown',
                    options=self.options['crop_types'],
                    value='fov_only',
                    style={'width': '100%'},
                    clearable=False
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginRight': '5%'}),
            
            # Anchor selection
            html.Div([
                html.Label(
                    "Anchor:", 
                    style={'fontWeight': 'bold', 'marginBottom': 10}
                ),
                dcc.Dropdown(
                    id='anchor-dropdown',
                    options=self.options['anchors'],
                    value='pos_x',
                    style={'width': '100%'},
                    clearable=False
                )
            ], style={'width': '30%', 'display': 'inline-block'})
        ], style={'marginBottom': 30})
    
    def create_main_plot(self) -> dcc.Loading:
        """Create the main 3D plot with loading indicator.
        
        Returns:
            Dash Loading component containing the main plot
        """
        return dcc.Loading(
            id="loading",
            type="default",
            children=[
                dcc.Graph(
                    id='main-3d-plot',
                    style={'height': '700px'}
                ),
            ]
        )
    
    def create_instructions(self) -> html.Div:
        """Create the instructions panel.
        
        Returns:
            Dash HTML component for instructions
        """
        return html.Div([
            html.H3("Instructions:"),
            html.Ul([
                html.Li("Use the dropdowns above to select different point clouds, crop types, and camera anchors"),
                html.Li("Blue points (transparent) = original points that were removed"),
                html.Li("Red points = points kept after cropping"),
                html.Li("Black diamond = sensor position"),
                html.Li("Green arrow = sensor viewing direction"),
                html.Li("Purple surface = range limit (for range-only cropping)"),
                html.Li("Orange lines = field of view boundaries (for FOV-only cropping)"),
                html.Li("Use mouse to rotate, zoom, and pan the 3D view")
            ]),
            html.Hr(),
            html.H4("Crop Type Details:"),
            html.Ul([
                html.Li([
                    html.Strong("Range Only"), 
                    ": Filters points based on distance from sensor (max 6m)"
                ]),
                html.Li([
                    html.Strong("FOV Only"), 
                    ": Filters points based on field-of-view cone (80° horizontal, ±20° vertical)"
                ]),
                html.Li([
                    html.Strong("Occlusion Only"), 
                    ": Filters points based on line-of-sight visibility from sensor"
                ])
            ]),
            html.Hr(),
            html.H4("Point Cloud Types:"),
            html.Ul([
                html.Li([
                    html.Strong("Cube"), 
                    ": 3000 points on surface of 4×4×4 cube - demonstrates geometric edge effects"
                ]),
                html.Li([
                    html.Strong("Sphere"), 
                    ": 2000 points with uniform volume distribution - shows smooth surface filtering"
                ]),
                html.Li([
                    html.Strong("Scene"), 
                    ": 4000 points representing complex outdoor environment with multiple objects"
                ])
            ])
        ], style={
            'marginTop': 20,
            'padding': 20,
            'backgroundColor': '#f0f0f0',
            'borderRadius': 5
        })
    
    def create_info_panel(self) -> html.Div:
        """Create dynamic info panel (populated by callbacks).
        
        Returns:
            Dash HTML component for info panel
        """
        return html.Div(
            id='info-panel',
            style={
                'marginTop': 20,
                'padding': 20,
                'backgroundColor': '#e8f4fd',
                'borderRadius': 5,
                'border': '1px solid #0066cc'
            }
        )
    
    def create_layout(self) -> html.Div:
        """Create the complete app layout.
        
        Returns:
            Complete Dash HTML layout
        """
        return html.Div([
            self.create_header(),
            self.create_controls(),
            self.create_main_plot(),
            self.create_info_panel(),
            self.create_instructions()
        ])
    
    def get_control_ids(self) -> Dict[str, str]:
        """Get the IDs of control components for callbacks.
        
        Returns:
            Dictionary mapping control names to their component IDs
        """
        return {
            'point_cloud': 'point-cloud-dropdown',
            'crop_type': 'crop-type-dropdown',
            'anchor': 'anchor-dropdown',
            'main_plot': 'main-3d-plot',
            'info_panel': 'info-panel'
        }