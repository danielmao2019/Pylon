#!/usr/bin/env python3
"""
Callback module for LiDAR simulation cropping visualization.
Handles all interactive functionality and updates.
"""

from dash import Input, Output, callback, html
import plotly.graph_objects as go
from typing import Tuple, List, Any

from .backend import LiDARVisualizationBackend


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
        
        @callback(
            [Output(self.control_ids['main_plot'], 'figure'),
             Output(self.control_ids['info_panel'], 'children')],
            [Input(self.control_ids['point_cloud'], 'value'),
             Input(self.control_ids['crop_type'], 'value'),
             Input(self.control_ids['anchor'], 'value')]
        )
        def update_visualization(cloud_name: str, crop_type: str, anchor: str) -> Tuple[go.Figure, List[Any]]:
            """Update the main 3D plot and info panel based on control selections.
            
            Args:
                cloud_name: Selected point cloud name
                crop_type: Selected crop type
                anchor: Selected anchor name
                
            Returns:
                Tuple of (figure, info_panel_children)
            """
            # Create the 3D plot
            fig = self.backend.create_3d_scatter_plot(cloud_name, crop_type, anchor)
            
            # Create info panel content
            info_content = self._create_info_content(cloud_name, crop_type, anchor)
            
            return fig, info_content
    
    def _create_info_content(self, cloud_name: str, crop_type: str, anchor: str) -> List[Any]:
        """Create content for the info panel.
        
        Args:
            cloud_name: Selected point cloud name
            crop_type: Selected crop type
            anchor: Selected anchor name
            
        Returns:
            List of Dash HTML components for info panel
        """
        try:
            # Get processed data for statistics
            data = self.backend.process_cropping(cloud_name, crop_type, anchor)
            
            original_count = len(data['original_points'])
            cropped_count = len(data['cropped_points'])
            reduction = data['reduction']
            main_pose = data['main_pose']
            anchor_poses = data['anchor_poses']
            crop_config = data['crop_config']
            
            # Create info content
            info_content = [
                html.H3(f"Current Configuration", style={'marginBottom': 15}),
                
                # Configuration summary
                html.Div([
                    html.Div([
                        html.Strong("Point Cloud: "),
                        cloud_name.title()
                    ], style={'marginBottom': 5}),
                    html.Div([
                        html.Strong("Crop Type: "),
                        crop_type.replace('_', ' ').title()
                    ], style={'marginBottom': 5}),
                    html.Div([
                        html.Strong("Anchor: "),
                        anchor.replace('_', ' ').title()
                    ], style={'marginBottom': 15})
                ]),
                
                # Statistics
                html.H4("Statistics", style={'marginBottom': 10}),
                html.Div([
                    html.Div([
                        html.Strong("Original Points: "),
                        f"{original_count:,}"
                    ], style={'marginBottom': 5}),
                    html.Div([
                        html.Strong("Points After Cropping: "),
                        f"{cropped_count:,}"
                    ], style={'marginBottom': 5}),
                    html.Div([
                        html.Strong("Reduction: "),
                        html.Span(
                            f"{reduction:.1f}%",
                            style={
                                'fontWeight': 'bold',
                                'color': 'red' if reduction > 75 else 'orange' if reduction > 50 else 'green'
                            }
                        )
                    ], style={'marginBottom': 5}),
                    html.Div([
                        html.Strong("Displayed Pose: "),
                        main_pose
                    ], style={'marginBottom': 5}),
                    html.Div([
                        html.Strong("Available Poses for Anchor: "),
                        f"{len(anchor_poses)}"
                    ], style={'marginBottom': 15})
                ]),
                
                # Crop configuration details
                html.H4("Crop Configuration", style={'marginBottom': 10}),
                self._create_crop_config_details(crop_config, crop_type),
                
                # Performance info
                html.Hr(),
                html.Div([
                    html.Small([
                        "💡 ",
                        html.Strong("Tip: "),
                        "Use your mouse to rotate, zoom, and pan the 3D visualization above. "
                        "Try different combinations of point clouds, crop types, and anchors to "
                        "explore how LiDAR filtering affects different geometries."
                    ], style={'color': '#666'})
                ])
            ]
            
        except Exception as e:
            # Error handling
            info_content = [
                html.H3("Error", style={'color': 'red'}),
                html.P(f"Failed to process configuration: {str(e)}"),
                html.P("Please try a different combination of settings.")
            ]
        
        return info_content
    
    def _create_crop_config_details(self, crop_config, crop_type: str) -> html.Div:
        """Create detailed configuration information.
        
        Args:
            crop_config: LiDAR crop configuration object
            crop_type: Type of cropping
            
        Returns:
            Dash HTML component with configuration details
        """
        details = []
        
        if crop_config.apply_range_filter:
            details.append(html.Li([
                html.Strong("Range Filter: "),
                f"Enabled (max distance: {crop_config.max_range:.1f}m)"
            ]))
        else:
            details.append(html.Li([
                html.Strong("Range Filter: "),
                "Disabled"
            ]))
        
        if crop_config.apply_fov_filter:
            h_fov = crop_config.horizontal_fov
            v_fov_min, v_fov_max = crop_config.vertical_fov
            details.append(html.Li([
                html.Strong("FOV Filter: "),
                f"Enabled (H: {h_fov:.0f}°, V: {v_fov_min:.0f}° to {v_fov_max:.0f}°)"
            ]))
        else:
            details.append(html.Li([
                html.Strong("FOV Filter: "),
                "Disabled"
            ]))
        
        if crop_config.apply_occlusion_filter:
            details.append(html.Li([
                html.Strong("Occlusion Filter: "),
                "Enabled (line-of-sight visibility)"
            ]))
        else:
            details.append(html.Li([
                html.Strong("Occlusion Filter: "),
                "Disabled"
            ]))
        
        # Add description based on crop type
        descriptions = {
            'range_only': "Only distance-based filtering is applied. Points beyond the maximum range are removed.",
            'fov_only': "Only field-of-view filtering is applied. Points outside the sensor's viewing cone are removed.",
            'occlusion_only': "Only occlusion filtering is applied. Points blocked by other points are removed."
        }
        
        return html.Div([
            html.Ul(details),
            html.P([
                html.Strong("Description: "),
                descriptions.get(crop_type, "Combined filtering approach.")
            ], style={'marginTop': 10, 'fontStyle': 'italic'})
        ])