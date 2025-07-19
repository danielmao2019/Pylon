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
                        style={'fontWeight': 'bold', 'marginBottom': 5}
                    ),
                    dcc.Dropdown(
                        id='point-cloud-dropdown',
                        options=self.options['point_clouds'],
                        value='cube',
                        style={'width': '100%'},
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
                
                # Crop type selection
                html.Div([
                    html.Label(
                        "Crop Type:", 
                        style={'fontWeight': 'bold', 'marginBottom': 5}
                    ),
                    dcc.Dropdown(
                        id='crop-type-dropdown',
                        options=self.options['crop_types'],
                        value='fov_only',
                        style={'width': '100%'},
                        clearable=False
                    )
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'marginBottom': 20}),
            
            # Camera pose controls
            html.H4("Camera Pose Controls", style={'marginBottom': 15}),
            self._create_camera_sliders(defaults),
            
            # Crop parameter controls
            html.H4("Crop Parameters", style={'marginBottom': 15, 'marginTop': 25}),
            self._create_crop_sliders(crop_defaults),
            
        ], style={'marginBottom': 30})
    
    def _create_camera_sliders(self, defaults: Dict[str, float]) -> html.Div:
        """Create camera pose slider controls.
        
        Args:
            defaults: Default camera pose values
            
        Returns:
            Dash HTML component with camera sliders
        """
        return html.Div([
            # Camera position sliders
            html.Div([
                html.H5("Camera Position", style={'marginBottom': 10}),
                # Azimuth slider
                html.Div([
                    html.Label("Azimuth (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='azimuth-slider',
                        min=0, max=360, step=5,
                        value=defaults['azimuth'],
                        marks={i: f"{i}°" for i in range(0, 361, 45)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': 15}),
                
                # Elevation slider
                html.Div([
                    html.Label("Elevation (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='elevation-slider',
                        min=-90, max=90, step=5,
                        value=defaults['elevation'],
                        marks={i: f"{i}°" for i in range(-90, 91, 30)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': 15}),
                
                # Distance slider
                html.Div([
                    html.Label("Distance: ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='distance-slider',
                        min=1, max=20, step=0.5,
                        value=defaults['distance'],
                        marks={i: f"{i}m" for i in range(1, 21, 3)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': 15})
            ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%'}),
            
            # Camera rotation sliders
            html.Div([
                html.H5("Camera Rotation", style={'marginBottom': 10}),
                # Yaw slider
                html.Div([
                    html.Label("Yaw (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='yaw-slider',
                        min=-180, max=180, step=5,
                        value=defaults['yaw'],
                        marks={i: f"{i}°" for i in range(-180, 181, 60)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': 15}),
                
                # Pitch slider
                html.Div([
                    html.Label("Pitch (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='pitch-slider',
                        min=-90, max=90, step=5,
                        value=defaults['pitch'],
                        marks={i: f"{i}°" for i in range(-90, 91, 30)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': 15}),
                
                # Roll slider
                html.Div([
                    html.Label("Roll (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='roll-slider',
                        min=-180, max=180, step=5,
                        value=defaults['roll'],
                        marks={i: f"{i}°" for i in range(-180, 181, 60)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': 15})
            ], style={'width': '48%', 'display': 'inline-block'})
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
                html.Label("Range Max (m): ", style={'display': 'inline-block', 'width': '30%'}),
                dcc.Slider(
                    id='range-max-slider',
                    min=1, max=15, step=0.5,
                    value=crop_defaults['range_only']['range_max'],
                    marks={i: f"{i}m" for i in range(1, 16, 2)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    disabled=True  # Initially disabled, enabled by callback based on crop type
                )
            ], style={'marginBottom': 15}),
            
            # FOV parameters (for fov_only)
            html.Div([
                # Horizontal FOV
                html.Div([
                    html.Label("Horizontal FOV (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='h-fov-slider',
                        min=10, max=180, step=5,
                        value=crop_defaults['fov_only']['h_fov'],
                        marks={i: f"{i}°" for i in range(10, 181, 30)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        disabled=False  # Initially enabled for fov_only default
                    )
                ], style={'marginBottom': 15}),
                
                # Vertical FOV span
                html.Div([
                    html.Label("Vertical FOV Span (°): ", style={'display': 'inline-block', 'width': '30%'}),
                    dcc.Slider(
                        id='v-fov-span-slider',
                        min=5, max=120, step=5,
                        value=crop_defaults['fov_only']['v_fov_span'],
                        marks={i: f"{i}°" for i in range(5, 121, 20)},
                        tooltip={"placement": "bottom", "always_visible": True},
                        disabled=False  # Initially enabled for fov_only default
                    )
                ], style={'marginBottom': 15})
            ])
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
                html.Li("Use the dropdowns to select different point clouds and crop types"),
                html.Li("Use the camera pose sliders to position and orient the sensor interactively"),
                html.Li("Use the crop parameter sliders to adjust filtering settings (enabled based on crop type)"),
                html.Li("Blue points (transparent) = original points that were removed"),
                html.Li("Red points = points kept after cropping"),
                html.Li("Black diamond = sensor position"),
                html.Li("Green arrow = sensor viewing direction"),
                html.Li("Purple surface = range limit (for range-only cropping)"),
                html.Li("Orange lines = field of view boundaries (for FOV-only cropping)"),
                html.Li("Use mouse to rotate, zoom, and pan the 3D view")
            ]),
            html.Hr(),
            html.H4("Camera Pose Controls:"),
            html.Ul([
                html.Li([
                    html.Strong("Azimuth"), 
                    ": Horizontal rotation around origin (0° = +X axis, 90° = +Y axis)"
                ]),
                html.Li([
                    html.Strong("Elevation"), 
                    ": Vertical angle above/below horizon (positive = above, negative = below)"
                ]),
                html.Li([
                    html.Strong("Distance"), 
                    ": Distance from origin to sensor position"
                ]),
                html.Li([
                    html.Strong("Yaw"), 
                    ": Sensor rotation around its Z-axis (relative to look-at-origin)"
                ]),
                html.Li([
                    html.Strong("Pitch"), 
                    ": Sensor rotation around its Y-axis (tilt up/down)"
                ]),
                html.Li([
                    html.Strong("Roll"), 
                    ": Sensor rotation around its X-axis (bank left/right)"
                ])
            ]),
            html.Hr(),
            html.H4("Crop Type Details:"),
            html.Ul([
                html.Li([
                    html.Strong("Range Only"), 
                    ": Filters points based on distance from sensor (adjustable max range)"
                ]),
                html.Li([
                    html.Strong("FOV Only"), 
                    ": Filters points based on field-of-view cone (adjustable horizontal and vertical FOV)"
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