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
        """Create the control panel with dropdowns and sliders.
        
        Returns:
            Dash HTML component for controls
        """
        defaults = self.backend.get_default_camera_pose()
        crop_defaults = self.backend.get_default_crop_params()
        
        return html.Div([
            # Top row: Point cloud and crop type selection
            html.Div([
                # Point cloud selection
                html.Div([
                    html.Label(
                        "Point Cloud:", 
                        style={'fontWeight': 'bold', 'marginBottom': 5, 'fontSize': '14px'}
                    ),
                    dcc.Dropdown(
                        id='point-cloud-dropdown',
                        options=self.options['point_clouds'],
                        value='cube',
                        style={'width': '100%', 'fontSize': '14px'},
                        clearable=False
                    )
                ], style={'marginBottom': 15}),
                
                # Crop type selection
                html.Div([
                    html.Label(
                        "Crop Type:", 
                        style={'fontWeight': 'bold', 'marginBottom': 5, 'fontSize': '14px'}
                    ),
                    dcc.Dropdown(
                        id='crop-type-dropdown',
                        options=self.options['crop_types'],
                        value='fov_only',
                        style={'width': '100%', 'fontSize': '14px'},
                        clearable=False
                    )
                ], style={'marginBottom': 20})
            ]),
            
            # Camera pose controls
            html.H4("Camera Pose", style={'marginBottom': 10, 'fontSize': '16px'}),
            self._create_camera_sliders(defaults),
            
            # Crop parameter controls
            html.H4("Crop Parameters", style={'marginBottom': 10, 'marginTop': 20, 'fontSize': '16px'}),
            self._create_crop_sliders(crop_defaults),
            
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})
    
    def _create_camera_sliders(self, defaults: Dict[str, float]) -> html.Div:
        """Create camera pose slider controls.
        
        Args:
            defaults: Default camera pose values
            
        Returns:
            Dash HTML component with camera sliders
        """
        return html.Div([
            # Position controls
            html.H5("Position", style={'marginBottom': 8, 'fontSize': '14px', 'color': '#666'}),
            # Azimuth slider
            html.Div([
                html.Label("Azimuth (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='azimuth-slider',
                    min=0, max=360, step=5,
                    value=defaults['azimuth'],
                    marks={i: f"{i}°" for i in range(0, 361, 90)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'marginBottom': 12}),
            
            # Elevation slider
            html.Div([
                html.Label("Elevation (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='elevation-slider',
                    min=-90, max=90, step=5,
                    value=defaults['elevation'],
                    marks={i: f"{i}°" for i in range(-90, 91, 45)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'marginBottom': 12}),
            
            # Distance slider
            html.Div([
                html.Label("Distance (m)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='distance-slider',
                    min=1, max=20, step=0.5,
                    value=defaults['distance'],
                    marks={i: f"{i}" for i in range(1, 21, 4)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'marginBottom': 20}),
            
            # Rotation controls
            html.H5("Rotation", style={'marginBottom': 8, 'fontSize': '14px', 'color': '#666'}),
            # Yaw slider
            html.Div([
                html.Label("Yaw (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='yaw-slider',
                    min=-180, max=180, step=5,
                    value=defaults['yaw'],
                    marks={i: f"{i}°" for i in range(-180, 181, 90)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'marginBottom': 12}),
            
            # Pitch slider
            html.Div([
                html.Label("Pitch (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='pitch-slider',
                    min=-90, max=90, step=5,
                    value=defaults['pitch'],
                    marks={i: f"{i}°" for i in range(-90, 91, 45)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'marginBottom': 12}),
            
            # Roll slider
            html.Div([
                html.Label("Roll (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='roll-slider',
                    min=-180, max=180, step=5,
                    value=defaults['roll'],
                    marks={i: f"{i}°" for i in range(-180, 181, 90)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag'
                )
            ], style={'marginBottom': 12})
        ])
    
    def _create_crop_sliders(self, crop_defaults: Dict[str, Dict[str, float]]) -> html.Div:
        """Create crop parameter slider controls.
        
        Args:
            crop_defaults: Default crop parameter values
            
        Returns:
            Dash HTML component with crop sliders
        """
        return html.Div([
            # Range parameter (for range_only)
            html.Div([
                html.Label("Range Max (m)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='range-max-slider',
                    min=1, max=15, step=0.5,
                    value=crop_defaults['range_only']['range_max'],
                    marks={i: f"{i}" for i in range(1, 16, 3)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=True,  # Initially disabled, enabled by callback based on crop type
                    updatemode='drag'
                )
            ], style={'marginBottom': 12}),
            
            # Horizontal FOV (for fov_only)
            html.Div([
                html.Label("Horizontal FOV (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='h-fov-slider',
                    min=10, max=180, step=5,
                    value=crop_defaults['fov_only']['h_fov'],
                    marks={i: f"{i}" for i in range(10, 181, 40)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=False,  # Initially enabled for fov_only default
                    updatemode='drag'
                )
            ], style={'marginBottom': 12}),
            
            # Vertical FOV span (for fov_only)
            html.Div([
                html.Label("Vertical FOV Span (°)", style={'fontSize': '12px', 'marginBottom': 3}),
                dcc.Slider(
                    id='v-fov-span-slider',
                    min=5, max=120, step=5,
                    value=crop_defaults['fov_only']['v_fov_span'],
                    marks={i: f"{i}" for i in range(5, 121, 25)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=False,  # Initially enabled for fov_only default
                    updatemode='drag'
                )
            ], style={'marginBottom': 12})
        ])
    
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
                    style={'height': '600px', 'width': '100%'}
                ),
            ]
        )
    
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
            
            # Main content area with controls on left, visualization on right
            html.Div([
                # Left side: Controls
                html.Div([
                    self.create_controls()
                ], style={
                    'width': '30%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'paddingRight': '20px',
                    'boxSizing': 'border-box'
                }),
                
                # Right side: Visualization and stats
                html.Div([
                    self.create_main_plot(),
                    self.create_info_panel()
                ], style={
                    'width': '70%',
                    'display': 'inline-block',
                    'verticalAlign': 'top',
                    'boxSizing': 'border-box'
                })
            ], style={'marginTop': 20, 'width': '100%'})
        ])
    
    def get_control_ids(self) -> Dict[str, str]:
        """Get the IDs of control components for callbacks.
        
        Returns:
            Dictionary mapping control names to their component IDs
        """
        return {
            'point_cloud': 'point-cloud-dropdown',
            'crop_type': 'crop-type-dropdown',
            'main_plot': 'main-3d-plot',
            'info_panel': 'info-panel',
            # Camera pose sliders
            'azimuth': 'azimuth-slider',
            'elevation': 'elevation-slider', 
            'distance': 'distance-slider',
            'yaw': 'yaw-slider',
            'pitch': 'pitch-slider',
            'roll': 'roll-slider',
            # Crop parameter sliders
            'range_max': 'range-max-slider',
            'h_fov': 'h-fov-slider',
            'v_fov_span': 'v-fov-span-slider'
        }