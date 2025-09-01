"""Base display class for point cloud registration datasets with built-in display methods.

This module provides the BasePCRDataset class that inherits from BaseDataset
and includes type-specific display methods for point cloud registration datasets.
"""
from typing import Dict, Any, Optional, List, Tuple, Union
import random
import numpy as np
import torch
from dash import html
import plotly.graph_objects as go
from utils.point_cloud_ops import apply_transform
from utils.point_cloud_ops.set_ops import pc_symmetric_difference
from utils.point_cloud_ops.set_ops.symmetric_difference import _normalize_points
from utils.point_cloud_ops.apply_transform import _normalize_transform
from data.viewer.utils.atomic_displays.point_cloud_display import create_point_cloud_display, get_point_cloud_display_stats, build_point_cloud_id
from data.viewer.utils.display_utils import DisplayStyles, ParallelFigureCreator, create_figure_grid
from data.viewer.utils.structure_validation import validate_pcr_structure
from data.datasets.base_dataset import BaseDataset


class BasePCRDataset(BaseDataset):
    """Base display class for point cloud registration datasets.
    
    This class provides the standard INPUT_NAMES, LABEL_NAMES, and display_datapoint
    method for point cloud registration datasets. Concrete dataset classes should inherit
    from this class to automatically get appropriate display functionality.
    
    Expected data structure:
    - inputs: {'src_pc': Dict, 'tgt_pc': Dict}
      OR: {'points': List[torch.Tensor], 'lengths'/'stack_lengths': List[torch.Tensor]} (batched format)
    - labels: {'transform': torch.Tensor}
    """
    
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    
    @staticmethod
    def create_union_visualization(
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        point_size: float = 2,
        point_opacity: float = 0.8,
        camera_state: Optional[Dict[str, Any]] = None,
        lod_type: str = "continuous",
        point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
        density_percentage: int = 100,
        axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        title: str = "Union (Transformed Source + Target)",
    ) -> go.Figure:
        """Create a visualization of the union of transformed source and target point clouds.

        Args:
            src_points: Transformed source point cloud [N, 3] or [1, N, 3]
            tgt_points: Target point cloud [M, 3] or [1, M, 3]
            point_size: Size of points in visualization
            point_opacity: Opacity of points in visualization
            camera_state: Optional dictionary containing camera position state
            lod_type: Type of LOD ("continuous", "discrete", or "none")
            point_cloud_id: Unique identifier for LOD caching
            density_percentage: Percentage of points to display when lod_type is "none" (1-100)
            axis_ranges: Optional dictionary containing axis ranges for consistent scaling
            title: Title for the visualization

        Returns:
            Plotly figure showing the union visualization
        """
        # Normalize points to unbatched format
        src_points_normalized = _normalize_points(src_points)
        tgt_points_normalized = _normalize_points(tgt_points)
        
        # Combine points
        union_points = torch.cat([src_points_normalized, tgt_points_normalized], dim=0)

        # Create colors for union (red for source, blue for target)
        src_colors = torch.zeros((len(src_points_normalized), 3), device=src_points_normalized.device)
        src_colors[:, 0] = 1.0  # Red for source
        tgt_colors = torch.zeros((len(tgt_points_normalized), 3), device=tgt_points_normalized.device)
        tgt_colors[:, 2] = 1.0  # Blue for target
        union_colors = torch.cat([src_colors, tgt_colors], dim=0)

        return create_point_cloud_display(
            points=union_points,
            colors=union_colors,
            title=title,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state,
            lod_type=lod_type,
            density_percentage=density_percentage,
            point_cloud_id=point_cloud_id,
            axis_ranges=axis_ranges,
        )

    @staticmethod
    def create_symmetric_difference_visualization(
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        radius: float = 0.05,
        point_size: float = 2,
        point_opacity: float = 0.8,
        camera_state: Optional[Dict[str, Any]] = None,
        lod_type: str = "continuous",
        point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
        density_percentage: int = 100,
        axis_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        title: str = "Symmetric Difference",
    ) -> go.Figure:
        """Create a visualization of the symmetric difference between transformed source and target point clouds.

        Args:
            src_points: Transformed source point cloud [N, 3] or [1, N, 3]
            tgt_points: Target point cloud [M, 3] or [1, M, 3]
            radius: Radius for computing symmetric difference
            point_size: Size of points in visualization
            point_opacity: Opacity of points in visualization
            camera_state: Optional dictionary containing camera position state
            lod_type: Type of LOD ("continuous", "discrete", or "none")
            point_cloud_id: Unique identifier for LOD caching
            density_percentage: Percentage of points to display when lod_type is "none" (1-100)
            axis_ranges: Optional dictionary containing axis ranges for consistent scaling
            title: Title for the visualization

        Returns:
            Plotly figure showing the symmetric difference visualization
        """
        # Normalize points to unbatched format
        src_points_normalized = _normalize_points(src_points)
        tgt_points_normalized = _normalize_points(tgt_points)
        
        # Find points in symmetric difference
        src_indices, tgt_indices = pc_symmetric_difference(src_points_normalized, tgt_points_normalized, radius)

        if len(src_indices) > 0 or len(tgt_indices) > 0:
            # Extract points in symmetric difference
            src_diff = src_points_normalized[src_indices]
            tgt_diff = tgt_points_normalized[tgt_indices]

            # Combine the points
            sym_diff_points = torch.cat([src_diff, tgt_diff], dim=0)

            # Create colors for symmetric difference (red for source, blue for target)
            src_colors = torch.zeros((len(src_indices), 3), device=src_diff.device)
            src_colors[:, 0] = 1.0  # Red for source
            tgt_colors = torch.zeros((len(tgt_indices), 3), device=tgt_diff.device)
            tgt_colors[:, 2] = 1.0  # Blue for target
            sym_diff_colors = torch.cat([src_colors, tgt_colors], dim=0)

            return create_point_cloud_display(
                points=sym_diff_points,
                colors=sym_diff_colors,
                title=title,
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
                lod_type=lod_type,
                density_percentage=density_percentage,
                point_cloud_id=point_cloud_id,
                axis_ranges=axis_ranges,
            )
        else:
            # If no symmetric difference, show empty point cloud
            return create_point_cloud_display(
                torch.zeros((1, 3), device=src_points_normalized.device),
                title=f"{title} (Empty)",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
                lod_type=lod_type,
                density_percentage=density_percentage,
                point_cloud_id=point_cloud_id,
                axis_ranges=axis_ranges,
            )

    @staticmethod
    def _compute_transform_info(transform: torch.Tensor) -> Dict[str, Any]:
        """Compute transform information including rotation angle and translation magnitude."""
        # Normalize transform to handle batched case
        transform_normalized = _normalize_transform(transform, torch.Tensor, target_device=transform.device, target_dtype=transform.dtype)
        
        # Compute rotation angle and translation magnitude
        rotation_matrix = transform_normalized[:3, :3]
        translation_vector = transform_normalized[:3, 3]

        # Compute rotation angle using the trace of the rotation matrix
        trace = torch.trace(rotation_matrix)
        rotation_angle = torch.acos((trace - 1) / 2) * 180 / np.pi  # Convert to degrees

        # Compute translation magnitude
        translation_magnitude = torch.norm(translation_vector)

        # Format the transformation matrix as a string
        transform_str = "Transform Matrix:\n"
        for i in range(4):
            row = [f"{transform_normalized[i, j]:.4f}" for j in range(4)]
            transform_str += "  ".join(row) + "\n"

        return {
            'transform_str': transform_str,
            'rotation_angle': rotation_angle,
            'translation_magnitude': translation_magnitude
        }

    @staticmethod
    def _create_transform_info_section(transform_info: Dict[str, Any]) -> html.Div:
        """Create transform information section."""
        return html.Div([
            html.H4("Transform Information:"),
            html.Pre(transform_info['transform_str']),
            html.P(f"Rotation Angle: {transform_info['rotation_angle']:.2f} degrees"),
            html.P(f"Translation Magnitude: {transform_info['translation_magnitude']:.4f}")
        ], style={'margin-top': '20px'})

    @staticmethod
    def _create_statistics_section(src_stats_children: Any, tgt_stats_children: Any) -> html.Div:
        """Create point cloud statistics section."""
        return html.Div([
            html.Div([
                html.H4("Source Point Cloud Statistics:"),
                html.Div(src_stats_children)
            ], style=DisplayStyles.GRID_ITEM_48_MARGIN),
            
            html.Div([
                html.H4("Target Point Cloud Statistics:"),
                html.Div(tgt_stats_children)
            ], style=DisplayStyles.GRID_ITEM_48_NO_MARGIN)
        ], style={'margin-top': '20px'})

    @staticmethod
    def _create_correspondence_stats_section(correspondences: torch.Tensor) -> html.Div:
        """Create correspondence statistics section."""
        num_correspondences = correspondences.shape[0]
        
        return html.Div([
            html.H4("Correspondence Statistics:"),
            html.Ul([
                html.Li(f"Number of correspondences: {num_correspondences}")
            ], style={'margin-left': '20px', 'margin-top': '5px'})
        ], style={'margin-top': '20px'})

    @staticmethod
    def _format_value(key: str, value: Any) -> str:
        """Format a value for display based on key name and value type."""
        if isinstance(value, list) and len(value) == 3 and all(isinstance(x, (int, float)) for x in value):
            # Handle 3D vectors with context-specific formatting
            if 'angle' in key.lower():
                return f"[{value[0]:.2f}°, {value[1]:.2f}°, {value[2]:.2f}°]"
            else:
                return f"[{value[0]:.4f}, {value[1]:.4f}, {value[2]:.4f}]"
        elif isinstance(value, float):
            return f"{value:.4f}"
        else:
            return str(value)

    @staticmethod
    def _dict_to_html_list(data: Dict[str, Any], key_name: str = None) -> html.Div:
        """Convert a dictionary to HTML list structure."""
        items = []
        
        if key_name:
            items.append(html.H5(f"{key_name}:", style={'margin-top': '15px', 'margin-bottom': '5px'}))
        
        list_items = []
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested dictionary - create sub-list
                items.append(BasePCRDataset._dict_to_html_list(value, key))
            else:
                # Simple key-value pair
                formatted_value = BasePCRDataset._format_value(key, value)
                
                # Special styling for overlap (important PCR metric)
                if key == 'overlap':
                    list_items.append(html.Li(f"{key}: {formatted_value}", 
                                            style={'font-weight': 'bold', 'color': '#2E86AB'}))
                else:
                    list_items.append(html.Li(f"{key}: {formatted_value}"))
        
        if list_items:
            items.append(html.Ul(list_items, style={'margin-left': '20px', 'margin-top': '5px'}))
        
        return html.Div(items)

    @staticmethod
    def _create_meta_info_section(meta_info: Dict[str, Any]) -> html.Div:
        """Create meta information section displaying dataset metadata."""
        if not meta_info:
            return html.Div([
                html.H4("Datapoint Meta Information:"),
                html.P("No meta information available")
            ], style={'margin-top': '20px'})
        
        # Convert the entire meta_info dict to HTML lists
        return html.Div([
            html.H4("Datapoint Meta Information:"),
            BasePCRDataset._dict_to_html_list(meta_info)
        ], style={'margin-top': '20px'})

    @staticmethod
    def create_correspondence_visualization(
        src_points: torch.Tensor,
        tgt_points: torch.Tensor,
        correspondences: torch.Tensor,
        point_size: float = 2,
        point_opacity: float = 0.8,
        camera_state: Optional[Dict[str, Any]] = None,
        lod_type: str = "continuous",
        density_percentage: int = 100,
        point_cloud_id: Optional[Union[str, Tuple[str, int, str]]] = None,
        title: str = "Point Cloud Correspondences",
    ) -> go.Figure:
        """Create a side-by-side dual view visualization of correspondences between source and target point clouds.

        Args:
            src_points: Source point cloud [N, 3] or [1, N, 3]
            tgt_points: Target point cloud [M, 3] or [1, M, 3]
            correspondences: Correspondence pairs [K, 2] where each row is (src_idx, tgt_idx)
            point_size: Size of points in visualization
            point_opacity: Opacity of points in visualization
            camera_state: Optional dictionary containing camera position state
            lod_type: Type of LOD ("continuous", "discrete", or "none")
            density_percentage: Percentage of points to display when lod_type is "none" (1-100)
            point_cloud_id: Unique identifier for LOD caching
            title: Title for the visualization

        Returns:
            Plotly figure showing the side-by-side correspondence visualization
        """
        # Normalize points to unbatched format
        src_points_normalized = _normalize_points(src_points)
        tgt_points_normalized = _normalize_points(tgt_points)
        
        src_points_np = src_points_normalized.cpu().numpy()
        tgt_points_np = tgt_points_normalized.cpu().numpy()
        correspondences_np = correspondences.cpu().numpy()

        # Calculate spatial bounds for proper side-by-side positioning
        src_bounds = {
            'x': [src_points_np[:, 0].min(), src_points_np[:, 0].max()],
            'y': [src_points_np[:, 1].min(), src_points_np[:, 1].max()], 
            'z': [src_points_np[:, 2].min(), src_points_np[:, 2].max()]
        }
        tgt_bounds = {
            'x': [tgt_points_np[:, 0].min(), tgt_points_np[:, 0].max()],
            'y': [tgt_points_np[:, 1].min(), tgt_points_np[:, 1].max()],
            'z': [tgt_points_np[:, 2].min(), tgt_points_np[:, 2].max()]
        }
        
        # Calculate offset to position target cloud to the right of source cloud
        src_width = src_bounds['x'][1] - src_bounds['x'][0]
        tgt_width = tgt_bounds['x'][1] - tgt_bounds['x'][0]
        gap = max(src_width, tgt_width) * 0.3  # 30% gap between clouds
        x_offset = src_bounds['x'][1] + gap - tgt_bounds['x'][0]
        
        # Offset target points for side-by-side layout
        tgt_points_offset = tgt_points_np.copy()
        tgt_points_offset[:, 0] += x_offset
        
        # Create figure
        fig = go.Figure()
        
        # Add source point cloud (left side)
        fig.add_trace(go.Scatter3d(
            x=src_points_np[:, 0],
            y=src_points_np[:, 1], 
            z=src_points_np[:, 2],
            mode='markers',
            marker=dict(size=point_size, color='blue', opacity=point_opacity),
            name='Source Points',
            showlegend=True
        ))
        
        # Add target point cloud (right side, offset)
        fig.add_trace(go.Scatter3d(
            x=tgt_points_offset[:, 0],
            y=tgt_points_offset[:, 1],
            z=tgt_points_offset[:, 2], 
            mode='markers',
            marker=dict(size=point_size, color='red', opacity=point_opacity),
            name='Target Points',
            showlegend=True
        ))
        
        # Highlight corresponding points with brighter colors and draw connection lines
        if len(correspondences_np) > 0:
            # Limit to reasonable number of correspondences for visibility
            max_correspondences = 50
            if len(correspondences_np) > max_correspondences:
                # Sample correspondences for visualization
                sample_indices = np.random.choice(len(correspondences_np), max_correspondences, replace=False)
                correspondences_display = correspondences_np[sample_indices]
            else:
                correspondences_display = correspondences_np
                
            # Extract corresponding points
            src_corr_indices = correspondences_display[:, 0].astype(int)
            tgt_corr_indices = correspondences_display[:, 1].astype(int)
            
            src_corr_points = src_points_np[src_corr_indices]
            tgt_corr_points_offset = tgt_points_offset[tgt_corr_indices]
            
            # Add highlighted corresponding points
            fig.add_trace(go.Scatter3d(
                x=src_corr_points[:, 0],
                y=src_corr_points[:, 1],
                z=src_corr_points[:, 2],
                mode='markers',
                marker=dict(size=point_size*1.5, color='cyan', opacity=1.0),
                name=f'Source Correspondences ({len(correspondences_display)})',
                showlegend=True
            ))
            
            fig.add_trace(go.Scatter3d(
                x=tgt_corr_points_offset[:, 0], 
                y=tgt_corr_points_offset[:, 1],
                z=tgt_corr_points_offset[:, 2],
                mode='markers', 
                marker=dict(size=point_size*1.5, color='yellow', opacity=1.0),
                name=f'Target Correspondences ({len(correspondences_display)})',
                showlegend=True
            ))
            
            # Add correspondence lines connecting the two sides
            for i in range(len(correspondences_display)):
                src_point = src_corr_points[i]
                tgt_point = tgt_corr_points_offset[i]
                
                fig.add_trace(go.Scatter3d(
                    x=[src_point[0], tgt_point[0]],
                    y=[src_point[1], tgt_point[1]], 
                    z=[src_point[2], tgt_point[2]],
                    mode='lines',
                    line=dict(color='green', width=2, dash='dash'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Update layout for proper 3D visualization
        fig.update_layout(
            title=f"{title} ({len(correspondences_np)} total correspondences)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                aspectmode='data'  # Maintain aspect ratio
            ),
            showlegend=True,
            width=1000,  # Wider to accommodate side-by-side layout
            height=600
        )
        
        # Apply camera state if provided
        if camera_state is not None:
            fig.update_layout(scene_camera=camera_state)

        return fig

    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> html.Div:
        """Display a point cloud registration datapoint.
        
        Args:
            datapoint: Dictionary containing inputs, labels, and meta_info from dataset
            class_labels: Optional dictionary mapping class indices to label names (unused for PCR)
            camera_state: Optional dictionary containing camera position state for 3D visualizations
            settings_3d: Optional dictionary containing 3D visualization settings
            
        Returns:
            html.Div: HTML layout for displaying this datapoint
        """
        # Validate inputs
        assert datapoint is not None, "datapoint must not be None"
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        
        # Validate structure and inputs (includes all basic validation)
        validate_pcr_structure(datapoint)
        
        inputs = datapoint['inputs']

        # Extract visualization settings
        point_size = 2
        point_opacity = 0.8
        sym_diff_radius = 0.05
        lod_type = "continuous"
        density_percentage = 100
        
        # Unpack 3D settings if provided
        if settings_3d is not None:
            assert isinstance(settings_3d, dict), f"settings_3d must be dict, got {type(settings_3d)}"
            point_size = settings_3d.get('point_size', point_size)
            point_opacity = settings_3d.get('point_opacity', point_opacity)
            sym_diff_radius = settings_3d.get('sym_diff_radius', sym_diff_radius)
            lod_type = settings_3d.get('lod_type', lod_type)
            density_percentage = settings_3d.get('density_percentage', density_percentage)

        # Extract point clouds
        src_xyz = inputs['src_pc']['pos']  # Source point cloud
        tgt_xyz = inputs['tgt_pc']['pos']  # Target point cloud

        # Extract RGB colors if available
        src_rgb = inputs['src_pc'].get('rgb')
        tgt_rgb = inputs['tgt_pc'].get('rgb')

        # Extract transform if available
        transform = datapoint['labels'].get('transform')
        if transform is None:
            transform = torch.eye(4)  # Default to identity transform if not provided

        # Apply transform to source point cloud
        src_pc_transformed = apply_transform(src_xyz, transform)

        # Compute unified axis ranges across all point clouds for consistent scaling
        all_points = [src_xyz, tgt_xyz, src_pc_transformed]
        x_coords = torch.cat([pc[:, 0] for pc in all_points])
        y_coords = torch.cat([pc[:, 1] for pc in all_points])
        z_coords = torch.cat([pc[:, 2] for pc in all_points])
        
        # Add small padding for better visualization
        padding = 0.05  # 5% padding
        x_range_unified = [x_coords.min().item(), x_coords.max().item()]
        y_range_unified = [y_coords.min().item(), y_coords.max().item()]
        z_range_unified = [z_coords.min().item(), z_coords.max().item()]
        
        # Apply padding
        x_pad = (x_range_unified[1] - x_range_unified[0]) * padding
        y_pad = (y_range_unified[1] - y_range_unified[0]) * padding
        z_pad = (z_range_unified[1] - z_range_unified[0]) * padding
        
        unified_axis_ranges = {
            'x': (x_range_unified[0] - x_pad, x_range_unified[1] + x_pad),
            'y': (y_range_unified[0] - y_pad, y_range_unified[1] + y_pad),
            'z': (z_range_unified[0] - z_pad, z_range_unified[1] + z_pad)
        }

        # Define figure creation tasks
        figure_tasks = [
            lambda: create_point_cloud_display(
                points=src_xyz,
                colors=src_rgb,
                title="Source Point Cloud",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
                lod_type=lod_type,
                density_percentage=density_percentage,
                point_cloud_id=build_point_cloud_id(datapoint, "source"),
                axis_ranges=unified_axis_ranges,
            ),
            lambda: create_point_cloud_display(
                points=tgt_xyz,
                colors=tgt_rgb,
                title="Target Point Cloud",
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
                lod_type=lod_type,
                density_percentage=density_percentage,
                point_cloud_id=build_point_cloud_id(datapoint, "target"),
                axis_ranges=unified_axis_ranges,
            ),
            lambda: BasePCRDataset.create_union_visualization(
                src_pc_transformed,
                tgt_xyz,
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
                lod_type=lod_type,
                point_cloud_id=build_point_cloud_id(datapoint, "union"),
                density_percentage=density_percentage,
                axis_ranges=unified_axis_ranges,
            ),
            lambda: BasePCRDataset.create_symmetric_difference_visualization(
                src_pc_transformed,
                tgt_xyz,
                radius=sym_diff_radius,
                point_size=point_size,
                point_opacity=point_opacity,
                camera_state=camera_state,
                lod_type=lod_type,
                point_cloud_id=build_point_cloud_id(datapoint, "sym_diff"),
                density_percentage=density_percentage,
                axis_ranges=unified_axis_ranges,
            ),
        ]

        # Add correspondence visualization if correspondences are available
        if 'correspondences' in inputs:
            correspondences = inputs['correspondences']
            figure_tasks.append(
                lambda: BasePCRDataset.create_correspondence_visualization(
                    src_pc_transformed,  # Use transformed source points for alignment visualization
                    tgt_xyz,
                    correspondences=correspondences,
                    point_size=point_size,
                    point_opacity=point_opacity,
                    camera_state=camera_state,
                    lod_type=lod_type,
                    density_percentage=density_percentage,
                    point_cloud_id=build_point_cloud_id(datapoint, "correspondences"),
                    title="Point Cloud Correspondences",
                )
            )

        # Create figures in parallel using centralized utility
        figure_creator = ParallelFigureCreator(max_workers=4, enable_timing=False)
        figures = figure_creator.create_figures_parallel(figure_tasks)

        # Compute transform information
        transform_info = BasePCRDataset._compute_transform_info(transform)
        
        # Get point cloud statistics
        src_stats_dict = get_point_cloud_display_stats(inputs['src_pc'])
        tgt_stats_dict = get_point_cloud_display_stats(inputs['tgt_pc'])
        
        # Convert statistics to HTML elements
        src_stats_children = BasePCRDataset._dict_to_html_list(src_stats_dict)
        tgt_stats_children = BasePCRDataset._dict_to_html_list(tgt_stats_dict)

        # Create layout using centralized utilities
        grid_items = create_figure_grid(figures, width_style="50%", height_style="520px")
        
        # Build layout sections
        layout_sections = [
            html.H3("Point Cloud Registration Visualization"),
            html.Div(grid_items, style=DisplayStyles.FLEX_WRAP),
            BasePCRDataset._create_transform_info_section(transform_info),
            BasePCRDataset._create_statistics_section(src_stats_children, tgt_stats_children)
        ]
        
        # Add correspondence statistics if correspondences are available
        if 'correspondences' in inputs:
            correspondences = inputs['correspondences']
            layout_sections.append(BasePCRDataset._create_correspondence_stats_section(correspondences))
        
        # Add meta info section last
        layout_sections.append(BasePCRDataset._create_meta_info_section(datapoint.get('meta_info', {})))
        
        return html.Div(layout_sections)
