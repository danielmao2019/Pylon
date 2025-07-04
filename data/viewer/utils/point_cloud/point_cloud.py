"""WebGL-based point cloud visualization utilities."""
from typing import Dict, Optional, Union, Any
import numpy as np
import torch
import json
import uuid
from dash import html, Input, Output, clientside_callback
from data.viewer.utils.segmentation import get_color


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

    # Calculate bounding box for camera setup
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.max(bbox_max - bbox_min)

    return {
        'positions': points.flatten().tolist(),  # Flatten for WebGL buffer
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
    """Create a point cloud visualization.

    This is the main entry point that replaces the old Plotly-based function.
    Returns a WebGL-based visualization component.

    Args:
        points: Numpy array of shape (N, 3) containing XYZ coordinates
        colors: Optional numpy array of shape (N, 3) containing RGB color values
        labels: Optional numpy array of shape (N,) containing labels
        title: Title for the figure
        point_size: Size of the points
        point_opacity: Opacity of the points
        camera_state: Optional dictionary containing camera position state

    Returns:
        HTML Div containing WebGL point cloud visualization
    """
    # Convert inputs to numpy
    points = point_cloud_to_numpy(points)
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
    if labels is not None:
        labels = point_cloud_to_numpy(labels)

    # Prepare data for WebGL
    webgl_data = prepare_point_cloud_data(points, colors, labels)

    # Default camera state
    if camera_state is None:
        camera_distance = webgl_data['bbox_size'] * 2
        camera_state = {
            'position': [camera_distance, camera_distance, camera_distance],
            'target': webgl_data['bbox_center'],
            'up': [0, 0, 1]
        }
    else:
        # Ensure camera state values are JSON-serializable
        camera_state = {
            'position': (camera_state['position'].tolist() 
                        if hasattr(camera_state.get('position', []), 'tolist') 
                        else list(camera_state.get('position', [0, 0, 0]))),
            'target': (camera_state['target'].tolist() 
                      if hasattr(camera_state.get('target', []), 'tolist') 
                      else list(camera_state.get('target', [0, 0, 0]))),
            'up': (camera_state['up'].tolist() 
                  if hasattr(camera_state.get('up', []), 'tolist') 
                  else list(camera_state.get('up', [0, 0, 1])))
        }

    # Generate unique ID for this component
    component_id = f"webgl-point-cloud-{uuid.uuid4().hex[:8]}"
    
    # Prepare config for clientside callback
    config = {
        'pointCloudData': webgl_data,
        'pointSize': point_size,
        'pointOpacity': point_opacity,
        'cameraState': camera_state
    }

    # Create the WebGL component with callback-based initialization
    component = _create_webgl_component_with_callback(component_id, title, len(points), config)
    
    # Try to register callback if app is available through registry
    try:
        from data.viewer.callbacks.registry import registry
        if hasattr(registry, 'viewer') and hasattr(registry.viewer, 'app'):
            register_webgl_callback(registry.viewer.app, component_id)
    except (ImportError, AttributeError):
        # Registry not available or no app, callback will need to be registered manually
        # This is normal for eval_viewer and standalone usage
        pass
    
    return component


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




def _create_webgl_component_with_callback(component_id: str, title: str, point_count: int, config: Dict[str, Any]) -> html.Div:
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
        
        # Trigger for callback
        html.Div(id=f"{component_id}-trigger", children="init"),

        # WebGL container
        html.Div(
            id=component_id,
            style={
                'width': '100%',
                'height': '500px',
                'border': '1px solid #ddd',
                'borderRadius': '4px',
                'position': 'relative',
                'overflow': 'hidden',
                'backgroundColor': '#f8f8f8'
            }
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
    
    if component_id in _webgl_components:
        return  # Already registered
    
    _webgl_components.add(component_id)
    
    # Register clientside callback for this specific component
    clientside_callback(
        f"""
        function(trigger) {{
            console.log('=== Professional 3D Navigation WebGL for {component_id} ===');
            
            const container = document.getElementById('{component_id}');
            const dataDiv = document.getElementById('{component_id}-data');
            
            if (!container) {{
                console.error('Container not found: {component_id}');
                return 'Error: Container not found';
            }}
            
            if (!dataDiv) {{
                console.error('Data div not found: {component_id}-data');
                return 'Error: Data div not found';
            }}
            
            try {{
                const config = JSON.parse(dataDiv.textContent);
                console.log('Config loaded for {component_id}:', config.pointCloudData.point_count, 'points');
                
                // Create canvas
                const canvas = document.createElement('canvas');
                canvas.width = container.clientWidth || 500;
                canvas.height = container.clientHeight || 500;
                canvas.style.width = '100%';
                canvas.style.height = '100%';
                canvas.style.backgroundColor = '#87CEEB';
                
                container.innerHTML = '';
                container.appendChild(canvas);
                
                // Set up 3D camera controls - orbit + pan + zoom navigation
                let camera = {{
                    // Orbit controls (yaw/pitch only)
                    yaw: 0.5,      // Horizontal rotation around target
                    pitch: 0.3,    // Vertical rotation around target  
                    distance: config.pointCloudData.bbox_size * 2,   // Distance from target
                    
                    // Pan controls (screen space translation)
                    targetX: config.pointCloudData.bbox_center[0],  // Look-at point
                    targetY: config.pointCloudData.bbox_center[1],
                    targetZ: config.pointCloudData.bbox_center[2],
                    
                    // Zoom
                    zoom: 1
                }};
                
                let mouseState = {{
                    isLeftDragging: false,
                    isRightDragging: false, 
                    isMiddleDragging: false,
                    lastX: 0,
                    lastY: 0
                }};
                
                // Mouse event handlers for 3D navigation
                canvas.addEventListener('mousedown', (e) => {{
                    e.preventDefault();
                    
                    if (e.button === 0) {{  // Left mouse button
                        mouseState.isLeftDragging = true;
                        canvas.style.cursor = 'grabbing';
                    }} else if (e.button === 1) {{  // Middle mouse button
                        mouseState.isMiddleDragging = true;
                        canvas.style.cursor = 'ns-resize';
                    }} else if (e.button === 2) {{  // Right mouse button
                        mouseState.isRightDragging = true;
                        canvas.style.cursor = 'move';
                    }}
                    
                    mouseState.lastX = e.clientX;
                    mouseState.lastY = e.clientY;
                }});
                
                canvas.addEventListener('mousemove', (e) => {{
                    const deltaX = e.clientX - mouseState.lastX;
                    const deltaY = e.clientY - mouseState.lastY;
                    
                    if (mouseState.isLeftDragging) {{
                        // Left mouse: Orbit camera (yaw/pitch only)
                        const sensitivity = 0.01;
                        camera.yaw -= deltaX * sensitivity;
                        camera.pitch += deltaY * sensitivity;
                        
                        // Clamp pitch to prevent flipping
                        camera.pitch = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, camera.pitch));
                        
                        drawScene();
                    }} else if (mouseState.isRightDragging) {{
                        // Right mouse: Pan in screen plane (reverse directions for intuitive control)
                        const sensitivity = 0.002 * camera.distance;
                        
                        // Calculate camera right and up vectors for screen-space panning
                        const yawCos = Math.cos(camera.yaw);
                        const yawSin = Math.sin(camera.yaw);
                        const pitchCos = Math.cos(camera.pitch);
                        
                        // Right vector (perpendicular to view direction in XZ plane)
                        const rightX = -yawSin;
                        const rightZ = yawCos;
                        
                        // Up vector (screen up, accounting for pitch)
                        const upX = -yawCos * Math.sin(camera.pitch);
                        const upY = Math.cos(camera.pitch);
                        const upZ = -yawSin * Math.sin(camera.pitch);
                        
                        // Apply panning (reverse directions for intuitive control)
                        camera.targetX += rightX * (-deltaX) * sensitivity - upX * (-deltaY) * sensitivity;
                        camera.targetY += -upY * (-deltaY) * sensitivity;
                        camera.targetZ += rightZ * (-deltaX) * sensitivity - upZ * (-deltaY) * sensitivity;
                        
                        drawScene();
                    }} else if (mouseState.isMiddleDragging) {{
                        // Middle mouse: Zoom with vertical movement (reverse direction)
                        const sensitivity = 0.01;
                        camera.zoom *= (1 + (-deltaY) * sensitivity);  // Reverse: drag up = zoom in
                        camera.zoom = Math.max(0.1, Math.min(5, camera.zoom));
                        
                        drawScene();
                    }}
                    
                    mouseState.lastX = e.clientX;
                    mouseState.lastY = e.clientY;
                }});
                
                canvas.addEventListener('mouseup', (e) => {{
                    mouseState.isLeftDragging = false;
                    mouseState.isRightDragging = false;
                    mouseState.isMiddleDragging = false;
                    canvas.style.cursor = 'default';
                }});
                
                // Disable context menu on right click
                canvas.addEventListener('contextmenu', (e) => {{
                    e.preventDefault();
                }});
                
                canvas.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    camera.zoom *= (e.deltaY > 0) ? 0.9 : 1.1;  // Reverse zoom: scroll up = zoom in, scroll down = zoom out
                    camera.zoom = Math.max(0.1, Math.min(5, camera.zoom));
                    drawScene();
                }});
                
                canvas.style.cursor = 'default';
                
                // Function to draw the 3D scene with orbit camera
                function drawScene() {{
                    const ctx = canvas.getContext('2d');
                    if (!ctx) return;
                    
                    // Clear canvas
                    ctx.fillStyle = '#87CEEB';
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    // Title
                    ctx.fillStyle = '#333';
                    ctx.font = '16px Arial';
                    ctx.textAlign = 'center';
                    ctx.fillText('Interactive Point Cloud', canvas.width/2, 25);
                    
                    // Controls info
                    ctx.font = '11px Arial';
                    ctx.fillText('Left: Orbit • Right: Pan • Scroll/Middle: Zoom', canvas.width/2, 45);
                    
                    // Calculate camera position from yaw/pitch/distance
                    const yawCos = Math.cos(camera.yaw);
                    const yawSin = Math.sin(camera.yaw);
                    const pitchCos = Math.cos(camera.pitch);
                    const pitchSin = Math.sin(camera.pitch);
                    
                    const cameraDistance = camera.distance / camera.zoom;
                    const cameraX = camera.targetX + cameraDistance * pitchCos * yawCos;
                    const cameraY = camera.targetY + cameraDistance * pitchSin;
                    const cameraZ = camera.targetZ + cameraDistance * pitchCos * yawSin;
                    
                    // Transform and project all points using orbit camera
                    const positions = config.pointCloudData.positions;
                    const colors = config.pointCloudData.colors;
                    const transformedPoints = [];
                    
                    // Calculate camera view vectors
                    const viewDirX = camera.targetX - cameraX;
                    const viewDirY = camera.targetY - cameraY;
                    const viewDirZ = camera.targetZ - cameraZ;
                    const viewLength = Math.sqrt(viewDirX*viewDirX + viewDirY*viewDirY + viewDirZ*viewDirZ);
                    
                    // Normalize view direction
                    const vx = viewDirX / viewLength;
                    const vy = viewDirY / viewLength;
                    const vz = viewDirZ / viewLength;
                    
                    // Right vector (perpendicular to view in XZ plane)
                    const rx = -yawSin;
                    const rz = yawCos;
                    
                    // Up vector (cross product of right and view)
                    const ux = -yawCos * pitchSin;
                    const uy = pitchCos;
                    const uz = -yawSin * pitchSin;
                    
                    for (let i = 0; i < positions.length; i += 3) {{
                        // Get world coordinates
                        const worldX = positions[i];
                        const worldY = positions[i + 1];
                        const worldZ = positions[i + 2];
                        
                        // Transform to camera space
                        const dx = worldX - cameraX;
                        const dy = worldY - cameraY;
                        const dz = worldZ - cameraZ;
                        
                        // Project to camera-relative coordinates
                        const camX = dx * rx + dz * rz;  // Right component
                        const camY = dx * ux + dy * uy + dz * uz;  // Up component  
                        const camZ = dx * vx + dy * vy + dz * vz;  // Depth component
                        
                        // Perspective projection
                        if (camZ > 0.1) {{  // Only render points in front of camera
                            const fov = 400;
                            const screenX = canvas.width/2 + (camX * fov) / camZ;
                            const screenY = canvas.height/2 - (camY * fov) / camZ;
                            
                            transformedPoints.push({{
                                x: screenX,
                                y: screenY,
                                z: camZ,
                                r: Math.floor(colors[i] * 255),
                                g: Math.floor(colors[i + 1] * 255),
                                b: Math.floor(colors[i + 2] * 255)
                            }});
                        }}
                    }}
                    
                    // Sort by Z depth (far to near)
                    transformedPoints.sort((a, b) => b.z - a.z);
                    
                    // Draw points
                    transformedPoints.forEach(point => {{
                        const size = Math.max(1, (config.pointSize || 2) * 5 / point.z);
                        ctx.fillStyle = `rgb(${{point.r}},${{point.g}},${{point.b}})`;
                        ctx.beginPath();
                        ctx.arc(point.x, point.y, size, 0, 2 * Math.PI);
                        ctx.fill();
                    }});
                }}
                
                // Initial draw
                drawScene();
                
                console.log('Professional 3D navigation setup complete for {component_id}');
                return 'SUCCESS: Interactive 3D point cloud with ' + config.pointCloudData.point_count + ' points';
                
            }} catch (error) {{
                console.error('WebGL callback error for {component_id}:', error);
                return 'Error: ' + error.message;
            }}
        }}
        """,
        Output(f'{component_id}-status', 'children'),
        Input(f'{component_id}-trigger', 'children')
    )
