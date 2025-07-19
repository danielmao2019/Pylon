#!/usr/bin/env python3
"""
Callback module for LiDAR simulation cropping visualization.
Handles all interactive functionality and updates.
"""

from dash import Input, Output, callback, html
import plotly.graph_objects as go
from typing import Tuple, List, Any, Dict

from demos.data.transforms.vision_3d.lidar_simulation_crop.webapp.backend import LiDARVisualizationBackend


class LiDARVisualizationCallbacks:
    """Callback class for handling web app interactivity."""
    
    def __init__(self, backend: LiDARVisualizationBackend, control_ids: dict):
        """Initialize callbacks with backend and control IDs.
        
        Args:
            backend: Backend instance for data processing
            control_ids: Dictionary of control component IDs
        """
        self.backend = backend
        self.control_ids = control_ids
    
    def register_callbacks(self) -> None:
        """Register all callback functions with the Dash app."""
        
        # Main visualization callback
        @callback(
            [Output(self.control_ids['main_plot'], 'figure'),
             Output(self.control_ids['info_panel'], 'children')],
            [Input(self.control_ids['point_cloud'], 'value'),
             Input(self.control_ids['crop_type'], 'value'),
             Input(self.control_ids['azimuth'], 'value'),
             Input(self.control_ids['elevation'], 'value'),
             Input(self.control_ids['distance'], 'value'),
             Input(self.control_ids['yaw'], 'value'),
             Input(self.control_ids['pitch'], 'value'),
             Input(self.control_ids['roll'], 'value'),
             Input(self.control_ids['range_max'], 'value'),
             Input(self.control_ids['h_fov'], 'value'),
             Input(self.control_ids['v_fov_span'], 'value')]
        )
        def update_visualization(cloud_name: str, crop_type: str, 
                               azimuth: float, elevation: float, distance: float,
                               yaw: float, pitch: float, roll: float,
                               range_max: float, h_fov: float, v_fov_span: float) -> Tuple[go.Figure, List[Any]]:
            """Update the main 3D plot and info panel based on control selections.
            
            Args:
                cloud_name: Selected point cloud name
                crop_type: Selected crop type
                azimuth: Camera azimuth angle
                elevation: Camera elevation angle
                distance: Camera distance from origin
                yaw: Camera yaw rotation
                pitch: Camera pitch rotation
                roll: Camera roll rotation
                range_max: Maximum range for range cropping
                h_fov: Horizontal FOV for FOV cropping
                v_fov_span: Vertical FOV span for FOV cropping
                
            Returns:
                Tuple of (figure, info_panel_children)
            """
            # Prepare crop parameters based on crop type
            crop_params = {}
            if crop_type == 'range_only':
                crop_params['range_max'] = range_max
            elif crop_type == 'fov_only':
                crop_params['h_fov'] = h_fov
                crop_params['v_fov_span'] = v_fov_span
            # occlusion_only doesn't need additional params
            
            # Create the 3D plot
            fig = self.backend.create_3d_scatter_plot(
                cloud_name, crop_type, azimuth, elevation, distance, 
                yaw, pitch, roll, **crop_params
            )
            
            # Create info panel content
            info_content = self._create_info_content(
                cloud_name, crop_type, azimuth, elevation, distance,
                yaw, pitch, roll, **crop_params
            )
            
            return fig, info_content
        
        # Crop parameter enable/disable callback
        @callback(
            [Output(self.control_ids['range_max'], 'disabled'),
             Output(self.control_ids['h_fov'], 'disabled'),
             Output(self.control_ids['v_fov_span'], 'disabled')],
            [Input(self.control_ids['crop_type'], 'value')]
        )
        def update_crop_slider_states(crop_type: str) -> Tuple[bool, bool, bool]:
            """Enable/disable crop parameter sliders based on crop type.
            
            Args:
                crop_type: Selected crop type
                
            Returns:
                Tuple of (range_disabled, h_fov_disabled, v_fov_span_disabled)
            """
            range_disabled = crop_type != 'range_only'
            fov_disabled = crop_type != 'fov_only'
            
            return range_disabled, fov_disabled, fov_disabled
    
    def _create_info_content(self, cloud_name: str, crop_type: str, 
                            azimuth: float, elevation: float, distance: float,
                            yaw: float, pitch: float, roll: float, **crop_params) -> List[Any]:
        """Create content for the info panel.
        
        Args:
            cloud_name: Selected point cloud name
            crop_type: Selected crop type
            azimuth: Camera azimuth angle
            elevation: Camera elevation angle
            distance: Camera distance from origin
            yaw: Camera yaw rotation
            pitch: Camera pitch rotation
            roll: Camera roll rotation
            **crop_params: Dynamic crop parameters
            
        Returns:
            List of Dash HTML components for info panel
        """
        try:
            # Get processed data for statistics
            data = self.backend.process_cropping(
                cloud_name, crop_type, azimuth, elevation, distance,
                yaw, pitch, roll, **crop_params
            )
            
            original_count = len(data['original_points'])
            cropped_count = len(data['cropped_points'])
            reduction = data['reduction']
            pose_description = data['pose_description']
            camera_params = data['camera_params']
            crop_config = data['crop_config']
            
            # Create info content
            info_content = [
                # Statistics
                html.H3("Statistics", style={'marginBottom': 15}),
                html.Div([
                    html.Div([
                        html.Strong("Point Cloud: "),
                        cloud_name.title(),
                        html.Span(" | ", style={'margin': '0 10px', 'color': '#999'}),
                        html.Strong("Crop Type: "),
                        crop_type.replace('_', ' ').title()
                    ], style={'marginBottom': 10}),
                    html.Div([
                        html.Strong("Original Points: "),
                        f"{original_count:,}",
                        html.Span(" → ", style={'margin': '0 10px', 'color': '#999'}),
                        html.Strong("After Cropping: "),
                        f"{cropped_count:,}",
                        html.Span(" (", style={'margin': '0 5px', 'color': '#999'}),
                        html.Span(
                            f"{reduction:.1f}% reduction",
                            style={
                                'fontWeight': 'bold',
                                'color': 'red' if reduction > 75 else 'orange' if reduction > 50 else 'green'
                            }
                        ),
                        html.Span(")", style={'color': '#999'})
                    ], style={'marginBottom': 15})
                ]),
                
                # Camera parameters details (compact)
                html.H4("Camera Pose", style={'marginBottom': 10}),
                html.Div([
                    html.Div([
                        html.Strong("Position: "),
                        f"Az={camera_params['azimuth']:.0f}°, El={camera_params['elevation']:.0f}°, Dist={camera_params['distance']:.1f}m"
                    ], style={'marginBottom': 5, 'fontSize': '14px'}),
                    html.Div([
                        html.Strong("Rotation: "),
                        f"Yaw={camera_params['yaw']:.0f}°, Pitch={camera_params['pitch']:.0f}°, Roll={camera_params['roll']:.0f}°"
                    ], style={'marginBottom': 15, 'fontSize': '14px'})
                ]),
                
                # Crop configuration details (compact)
                html.H4("Active Filters", style={'marginBottom': 10}),
                self._create_crop_config_details(crop_config, crop_type, crop_params)
            ]
            
        except Exception as e:
            # Error handling
            info_content = [
                html.H3("Error", style={'color': 'red'}),
                html.P(f"Failed to process configuration: {str(e)}"),
                html.P("Please try a different combination of settings.")
            ]
        
        return info_content
    
    def _create_crop_config_details(self, crop_config, crop_type: str, crop_params: Dict[str, Any]) -> html.Div:
        """Create detailed configuration information.
        
        Args:
            crop_config: LiDAR crop configuration object
            crop_type: Type of cropping
            crop_params: Current crop parameter values
            
        Returns:
            Dash HTML component with configuration details
        """
        # Create compact filter description
        active_filters = []
        
        if crop_config.apply_range_filter:
            range_val = crop_params.get('range_max', crop_config.max_range)
            active_filters.append(f"Range ≤ {range_val:.1f}m")
        
        if crop_config.apply_fov_filter:
            h_fov = crop_params.get('h_fov', crop_config.horizontal_fov)
            v_fov_span = crop_params.get('v_fov_span', 
                abs(crop_config.vertical_fov[1] - crop_config.vertical_fov[0]))
            active_filters.append(f"FOV {h_fov:.0f}°×{v_fov_span:.0f}°")
        
        if crop_config.apply_occlusion_filter:
            active_filters.append("Occlusion filtering")
        
        if not active_filters:
            active_filters = ["No filtering applied"]
        
        return html.Div([
            html.Div([
                html.Strong("Active: "),
                " | ".join(active_filters)
            ], style={'fontSize': '14px', 'marginBottom': 10}),
            
            # Add brief description for current crop type
            html.Div([
                html.Em(self._get_crop_description(crop_type))
            ], style={'fontSize': '13px', 'color': '#666'})
        ])
    
    def _get_crop_description(self, crop_type: str) -> str:
        """Get brief description for crop type.
        
        Args:
            crop_type: Type of cropping
            
        Returns:
            Brief description string
        """
        descriptions = {
            'range_only': "Distance-based filtering removes points beyond maximum range",
            'fov_only': "Field-of-view filtering removes points outside sensor cone",
            'occlusion_only': "Occlusion filtering removes points blocked by other points"
        }
        return descriptions.get(crop_type, "Combined filtering approach")