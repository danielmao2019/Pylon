"""WebGL-based point cloud visualization utilities."""
from typing import Dict, Optional, Union, Any
import numpy as np
import torch
import json
import uuid
from dash import html, Input, Output, clientside_callback, dcc
import plotly.graph_objects as go
from data.viewer.utils.segmentation import get_color

# Global registry for tracking created graphs for camera sync
_created_graphs = []


def point_cloud_to_numpy(points: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable point cloud."""
    if isinstance(points, torch.Tensor):
        return points.cpu().numpy()
    return points


def prepare_point_cloud_data(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Prepare point cloud data for WebGL rendering."""

    # Ensure points are float32 for WebGL
    points = points.astype(np.float32)

    # Handle colors
    if colors is not None:
        colors = colors.astype(np.float32)
        # Normalize colors to [0, 1] if they're in [0, 255] range
        if colors.max() > 1.0:
            colors = colors / 255.0
    elif labels is not None:
        # Convert labels to colors
        unique_labels = np.unique(labels)
        colors = np.zeros((len(points), 3), dtype=np.float32)

        for label in unique_labels:
            mask = labels == label
            color_hex = get_color(label)
            # Convert hex color to RGB
            r = int(color_hex[1:3], 16) / 255.0
            g = int(color_hex[3:5], 16) / 255.0
            b = int(color_hex[5:7], 16) / 255.0
            colors[mask] = [r, g, b]
    else:
        # Default blue color
        colors = np.full((len(points), 3), [0.27, 0.51, 0.71], dtype=np.float32)  # steelblue

    # Transform points to match working example coordinate range [-0.5, 0.5]
    # Working example generates points in this range directly
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.max(bbox_max - bbox_min)
    
    # Transform to [-0.5, 0.5] range like working example
    points_transformed = (points - bbox_center) / bbox_size

    return {
        'positions': points_transformed.flatten().tolist(),  # Transform to working range
        'colors': colors.flatten().tolist(),
        'point_count': len(points),
        'bbox_min': bbox_min.tolist(),
        'bbox_max': bbox_max.tolist(),
        'bbox_center': bbox_center.tolist(),
        'bbox_size': float(bbox_size)
    }




def create_point_cloud_figure(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None,
) -> html.Div:
    """Create a point cloud visualization with WebGL rendering and synchronized cameras.

    This function creates WebGL-based point cloud visualization with proper camera synchronization.
    Since JavaScript execution is blocked in this environment, it uses Plotly server-side rendering
    with synchronized camera states to achieve the desired behavior.

    Args:
        points: Numpy array of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N, 3) containing RGB color values
        labels: Optional numpy array of shape (N,) containing labels
        title: Title for the figure
        point_size: Size of the points
        point_opacity: Opacity of the points
        camera_state: Optional dictionary containing camera position state

    Returns:
        HTML Div containing synchronized point cloud visualization
    """
    print(f"=== CREATE_POINT_CLOUD_FIGURE CALLED ===")
    print(f"Title: {title}")
    print(f"Points type: {type(points)}")
    print(f"Points shape: {points.shape if hasattr(points, 'shape') else 'No shape'}")
    print(f"Colors type: {type(colors) if colors is not None else 'None'}")
    print(f"Point size: {point_size}, Opacity: {point_opacity}")
    
    # Convert inputs to numpy
    points = point_cloud_to_numpy(points)
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
    if labels is not None:
        labels = point_cloud_to_numpy(labels)

    print(f"=== AFTER NUMPY CONVERSION ===")
    print(f"Points shape: {points.shape}")
    print(f"Colors shape: {colors.shape if colors is not None else 'None'}")

    # Prepare data for rendering
    webgl_data = prepare_point_cloud_data(points, colors, labels)
    print(f"=== WEBGL DATA PREPARED ===")
    print(f"Point count: {webgl_data['point_count']}")
    print(f"Bbox size: {webgl_data['bbox_size']}")

    # Use consistent graph indices based on title to avoid mixing
    graph_index_map = {
        "Source Point Cloud": 0,
        "Target Point Cloud": 1,
        "Union (Transformed Source + Target)": 2,
        "Symmetric Difference": 3,
        "Symmetric Difference (Empty)": 3  # Same index as regular symmetric difference
    }
    
    # Get graph index from title, default to auto-increment for unknown titles
    if title in graph_index_map:
        graph_index = graph_index_map[title]
    else:
        # For other titles, use auto-increment
        if not hasattr(create_point_cloud_figure, '_graph_counter'):
            create_point_cloud_figure._graph_counter = 4  # Start after known indices
        graph_index = create_point_cloud_figure._graph_counter
        create_point_cloud_figure._graph_counter += 1
    
    # Create synchronized Plotly visualization (WebGL rendering via Plotly's WebGL backend)
    component = _create_plotly_point_cloud(
        webgl_data, 
        title, 
        point_size, 
        point_opacity,
        graph_index=graph_index,
        camera_state=camera_state
    )
    
    print(f"=== COMPONENT CREATED SUCCESSFULLY ===")
    return component


def _create_plotly_point_cloud(point_cloud_data: Dict[str, Any], title: str, point_size: float, point_opacity: float, graph_index: int = 0, camera_state: Optional[Dict[str, Any]] = None) -> dcc.Graph:
    """Create a server-side rendered point cloud using Plotly (no JavaScript required).
    
    Args:
        point_cloud_data: Prepared point cloud data from prepare_point_cloud_data()
        title: Title for the visualization
        point_size: Size of the points
        point_opacity: Opacity of the points
        graph_index: Index for camera synchronization
        camera_state: Optional camera state for synchronization
        
    Returns:
        Plotly Graph component with synchronized camera
    """
    print(f"=== CREATING PLOTLY POINT CLOUD: {title} ===")
    
    # Extract data
    positions = np.array(point_cloud_data['positions']).reshape(-1, 3)
    colors = np.array(point_cloud_data['colors']).reshape(-1, 3)
    point_count = point_cloud_data['point_count']
    
    print(f"Plotly positions shape: {positions.shape}")
    print(f"Plotly colors shape: {colors.shape}")
    print(f"Point count: {point_count}")
    
    # Convert colors to RGB strings
    color_strings = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in colors]
    
    # Create 3D scatter plot with WebGL acceleration
    fig = go.Figure(data=[go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1], 
        z=positions[:, 2],
        mode='markers',
        marker=dict(
            size=max(2, point_size),
            color=color_strings,
            opacity=point_opacity,
        ),
        text=[f'Point {i}' for i in range(len(positions))],
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    )])
    
    # Configure camera - convert from legacy format to Plotly format if needed
    default_camera = dict(
        eye=dict(x=1.5, y=1.5, z=1.5),
        center=dict(x=0, y=0, z=0), 
        up=dict(x=0, y=0, z=1)
    )
    
    # Convert camera_state from callback format to Plotly format
    if camera_state and isinstance(camera_state, dict):
        if 'eye' in camera_state:
            # Already in Plotly format
            camera_config = camera_state
            print(f"=== USING PLOTLY CAMERA: {camera_state['eye']} ===")
        elif 'position' in camera_state and 'target' in camera_state:
            # Convert from legacy position/target format to Plotly eye/center format
            camera_config = dict(
                eye=dict(x=camera_state['position'][0], y=camera_state['position'][1], z=camera_state['position'][2]),
                center=dict(x=camera_state['target'][0], y=camera_state['target'][1], z=camera_state['target'][2]),
                up=dict(x=0, y=0, z=1)  # Default up vector
            )
            print(f"=== CONVERTED LEGACY CAMERA TO PLOTLY: {camera_config['eye']} ===")
        else:
            camera_config = default_camera
            print(f"=== UNKNOWN CAMERA FORMAT, USING DEFAULT: {camera_config} ===")
    else:
        camera_config = default_camera
        print(f"=== NO CAMERA STATE, USING DEFAULT: {camera_config} ===")
    
    # Configure layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            bgcolor='lightblue',  # Light blue background
            camera=camera_config
        ),
        width=None,  # Auto-size to container
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    print(f"Plotly figure created for {title} with {point_count} points")
    
    # Register this graph for camera sync 
    _created_graphs.append(graph_index)
    
    return dcc.Graph(
        id={'type': 'point-cloud-graph', 'index': graph_index},  # Use proper pattern matching ID
        figure=fig,
        style={'height': '500px', 'width': '100%'},
        config={
            'displayModeBar': True, 
            'displaylogo': False,
            'plotGlPixelRatio': 2,  # High DPI support
            'toImageButtonOptions': {'format': 'png', 'filename': f'point_cloud_{title}', 'height': 500, 'width': 700, 'scale': 1}
        }  # Enable WebGL rendering and controls
    )


def get_point_cloud_stats(
    points: torch.Tensor,
    change_map: Optional[torch.Tensor] = None,
    class_names: Optional[Dict[int, str]] = None
) -> html.Ul:
    """Get statistical information about a point cloud.

    Args:
        points: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        List of html components with point cloud statistics
    """
    # Basic stats
    points_np = points.detach().cpu().numpy()
    stats_items = [
        html.Li(f"Total Points: {len(points_np):,}"),
        html.Li(f"Dimensions: {points_np.shape[1]}"),
        html.Li(f"X Range: [{points_np[:, 0].min():.2f}, {points_np[:, 0].max():.2f}]"),
        html.Li(f"Y Range: [{points_np[:, 1].min():.2f}, {points_np[:, 1].max():.2f}]"),
        html.Li(f"Z Range: [{points_np[:, 2].min():.2f}, {points_np[:, 2].max():.2f}]"),
        html.Li(f"Center: [{points_np[:, 0].mean():.2f}, {points_np[:, 1].mean():.2f}, {points_np[:, 2].mean():.2f}]")
    ]

    # Add class distribution if change_map is provided
    if change_map is not None:
        unique_classes, class_counts = torch.unique(change_map, return_counts=True)
        unique_classes = unique_classes.cpu().numpy()
        class_counts = class_counts.cpu().numpy()
        total_points = change_map.numel()

        stats_items.append(html.Li("Class Distribution:"))
        class_list_items = []

        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / total_points) * 100
            cls_key = cls.item() if hasattr(cls, 'item') else cls
            class_name = class_names[cls_key] if class_names and cls_key in class_names else f"Class {cls_key}"
            class_list_items.append(
                html.Li(f"{class_name}: {count:,} points ({percentage:.2f}%)",
                       style={'marginLeft': '20px'})
            )

        stats_items.append(html.Ul(class_list_items))

    return html.Ul(stats_items)




def _create_webgl_component_with_callback(component_id: str, title: str, point_count: int, config: Dict[str, Any], graph_index: int = 0) -> html.Div:
    """Create WebGL component using clientside callback for initialization."""
    
    return html.Div([
        html.H4(title, style={'marginBottom': '10px'}),

        # Performance info
        html.Div([
            html.Span(f"Displaying {point_count:,} points"),
            html.Span(f" • Point size: {config['pointSize']} • Opacity: {config['pointOpacity']}")
        ], style={'fontSize': '12px', 'color': '#666', 'marginBottom': '5px'}),

        # Hidden data storage for callback
        html.Div(id=f"{component_id}-data", 
                children=json.dumps(config),
                style={'display': 'none'}),
        

        # Server-side rendered point cloud using Plotly (NO JAVASCRIPT REQUIRED)
        _create_plotly_point_cloud(
            config['pointCloudData'], 
            title, 
            config['pointSize'], 
            config['pointOpacity'],
            graph_index=graph_index,
            camera_state=config.get('cameraState')
        ),

        # Status display
        html.Div(id=f"{component_id}-status",
                style={'fontSize': '11px', 'color': '#666', 'marginTop': '5px'}),

        # Controls info
        html.Div([
            html.Strong("Controls: "),
            html.Span("Left click + drag: Rotate • Right click + drag: Pan • Scroll: Zoom • R: Reset view • +/-: Point size")
        ], style={
            'fontSize': '11px',
            'color': '#888',
            'marginTop': '5px',
            'padding': '5px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '3px'
        })
    ])


# Global registry to track components and avoid duplicate callbacks
_webgl_components = set()

def register_webgl_callback(app, component_id: str):
    """Register a clientside callback for WebGL initialization."""
    
    print(f"register_webgl_callback called for {component_id}")
    print(f"App object: {app}")
    print(f"Component already registered: {component_id in _webgl_components}")
    
    if component_id in _webgl_components:
        print(f"Component {component_id} already registered, skipping")
        return  # Already registered
    
    print(f"Adding {component_id} to registered components")
    _webgl_components.add(component_id)
    
    # Register clientside callback for this specific component
    clientside_callback(
        f"""
        function(trigger) {{
            console.log('🚀 === WebGL Callback Triggered for {component_id} ===');
            console.log('🔍 Trigger value:', trigger);
            
            // Add delay to ensure DOM is ready
            setTimeout(function() {{
                console.log('⏰ Delayed execution starting for {component_id}');
                
                const container = document.getElementById('{component_id}');
                console.log('📦 Container found:', container);
                
                if (!container) {{
                    console.error('❌ Container not found: {component_id}');
                    return 'Error: Container not found';
                }}
                
                try {{
                    console.log('🎨 Creating canvas for {component_id}');
                    
                    // Create canvas
                    const canvas = document.createElement('canvas');
                    canvas.width = container.clientWidth || 500;
                    canvas.height = container.clientHeight || 500;
                    canvas.style.width = '100%';
                    canvas.style.height = '100%';
                    
                    console.log('📐 Canvas dimensions:', canvas.width, 'x', canvas.height);
                    
                    container.innerHTML = '';
                    container.appendChild(canvas);
                    console.log('🔗 Canvas appended to container');
                    
                    // Get WebGL context
                    const gl = canvas.getContext('webgl');
                    console.log('🎮 WebGL context:', gl);
                    
                    if (!gl) {{
                        console.error('❌ WebGL not supported');
                        return 'Error: WebGL not supported in this browser';
                    }}
                    
                    console.log('🌍 WebGL version:', gl.getParameter(gl.VERSION));
                    
                    // Set sky blue background and clear
                    gl.clearColor(0.53, 0.81, 0.98, 1.0); // Sky blue
                    console.log('🎨 Clear color set to sky blue');
                    
                    gl.clear(gl.COLOR_BUFFER_BIT);
                    console.log('🧹 Buffer cleared');
                    
                    gl.flush();
                    console.log('💫 GL flush executed');
                    
                    console.log('✅ Sky blue background rendered successfully for {component_id}');
                    
                }} catch (error) {{
                    console.error('💥 WebGL error for {component_id}:', error);
                    console.error('📋 Error stack:', error.stack);
                }}
            }}, 100);
            
            return '🔄 WebGL initialization in progress...';
        }}
        """,
        Output(f'{component_id}-status', 'children'),
        Input(f'{component_id}', 'id'),
        prevent_initial_call=False
    )
    
    print(f"Clientside callback registered successfully for {component_id}")
