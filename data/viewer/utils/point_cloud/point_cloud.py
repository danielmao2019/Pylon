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
        
        # Trigger for callback (hidden)
        html.Div(id=f"{component_id}-trigger", children="init", style={'display': 'none'}),

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
            console.log('=== Professional 3D WebGL Viewer for {component_id} ===');
            
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
                
                container.innerHTML = '';
                container.appendChild(canvas);
                
                // ===== WEBGL INITIALIZATION =====
                const gl = canvas.getContext('webgl');
                
                if (!gl) {{
                    console.error('WebGL not supported');
                    return 'Error: WebGL not supported in this browser';
                }}
                
                // ===== SHADER SOURCES =====
                const vertexShaderSource = `
                    attribute vec3 aPosition;
                    attribute vec3 aColor;
                    uniform mat3 uRotationMatrix;
                    uniform vec2 uTranslation;
                    uniform float uDepth;
                    varying vec3 vColor;
                    
                    void main() {{
                        // Apply world-space rotation matrix (EXACT COPY FROM WORKING EXAMPLE)
                        vec3 rotated = uRotationMatrix * aPosition;
                        
                        // Apply screen-space translation
                        rotated.x += uTranslation.x;
                        rotated.y += uTranslation.y;
                        
                        // Perspective projection with adjustable camera distance
                        float z = rotated.z + uDepth;
                        gl_Position = vec4(rotated.x / z, rotated.y / z, 0.0, 1.0);
                        gl_PointSize = 8.0;
                        vColor = aColor;
                    }}
                `;
                
                const fragmentShaderSource = `
                    precision mediump float;
                    varying vec3 vColor;
                    void main() {{
                        gl_FragColor = vec4(vColor, 1.0);
                    }}
                `;
                
                // ===== SHADER COMPILATION =====
                function createShader(type, source) {{
                    const shader = gl.createShader(type);
                    gl.shaderSource(shader, source);
                    gl.compileShader(shader);
                    
                    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
                        const error = gl.getShaderInfoLog(shader);
                        gl.deleteShader(shader);
                        throw new Error('Shader compilation error: ' + error);
                    }}
                    
                    return shader;
                }}
                
                let program;
                try {{
                    const vertexShader = createShader(gl.VERTEX_SHADER, vertexShaderSource);
                    const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
                    
                    // Create and link program
                    program = gl.createProgram();
                    gl.attachShader(program, vertexShader);
                    gl.attachShader(program, fragmentShader);
                    gl.linkProgram(program);
                    
                    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
                        throw new Error('Program linking error: ' + gl.getProgramInfoLog(program));
                    }}
                    
                    // Clean up shaders
                    gl.deleteShader(vertexShader);
                    gl.deleteShader(fragmentShader);
                }} catch (error) {{
                    console.error('Shader error:', error);
                    return 'Shader Error: ' + error.message;
                }}
                
                // ===== BUFFER CREATION =====
                const positions = config.pointCloudData.positions;
                const colors = config.pointCloudData.colors;
                const pointCount = config.pointCloudData.point_count;
                
                const positionBuffer = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
                
                const colorBuffer = gl.createBuffer();
                gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
                gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
                
                // ===== SHADER LOCATIONS =====
                const locations = {{
                    attributes: {{
                        position: gl.getAttribLocation(program, 'aPosition'),
                        color: gl.getAttribLocation(program, 'aColor')
                    }},
                    uniforms: {{
                        rotationMatrix: gl.getUniformLocation(program, 'uRotationMatrix'),
                        translation: gl.getUniformLocation(program, 'uTranslation'),
                        depth: gl.getUniformLocation(program, 'uDepth')
                    }}
                }};
                
                // ===== NAVIGATION STATE =====
                const navigation = {{
                    // World-space rotation matrix (identity initially)
                    rotationMatrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    
                    // Screen-space translation
                    translation: [0, 0],
                    
                    // Camera distance (match working example exactly)
                    depth: 3.0,
                    
                    // Sensitivity settings
                    sensitivity: {{
                        rotation: 0.01,
                        translation: 0.002,
                        depth: 0.1
                    }}
                }};
                
                // Mouse interaction state
                const mouse = {{
                    isLeftDragging: false,
                    isRightDragging: false,
                    lastX: 0,
                    lastY: 0
                }};
                
                // ===== MATRIX UTILITIES =====
                const MatrixUtils = {{
                    // Create rotation matrix around arbitrary axis using Rodrigues' formula
                    createRotation(axisX, axisY, axisZ, angle) {{
                        const c = Math.cos(angle);
                        const s = Math.sin(angle);
                        const t = 1 - c;
                        
                        return [
                            t*axisX*axisX + c,       t*axisX*axisY - s*axisZ, t*axisX*axisZ + s*axisY,
                            t*axisX*axisY + s*axisZ, t*axisY*axisY + c,       t*axisY*axisZ - s*axisX,
                            t*axisX*axisZ - s*axisY, t*axisY*axisZ + s*axisX, t*axisZ*axisZ + c
                        ];
                    }},
                    
                    // Multiply two 3x3 matrices: result = a * b
                    multiply(a, b) {{
                        return [
                            a[0]*b[0] + a[1]*b[3] + a[2]*b[6], a[0]*b[1] + a[1]*b[4] + a[2]*b[7], a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
                            a[3]*b[0] + a[4]*b[3] + a[5]*b[6], a[3]*b[1] + a[4]*b[4] + a[5]*b[7], a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
                            a[6]*b[0] + a[7]*b[3] + a[8]*b[6], a[6]*b[1] + a[7]*b[4] + a[8]*b[7], a[6]*b[2] + a[7]*b[5] + a[8]*b[8]
                        ];
                    }}
                }};
                
                // ===== RENDERING SYSTEM =====
                const RenderSystem = {{
                    // One-time setup of vertex attributes
                    setupVertexAttributes() {{
                        // Position attribute
                        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
                        gl.enableVertexAttribArray(locations.attributes.position);
                        gl.vertexAttribPointer(locations.attributes.position, 3, gl.FLOAT, false, 0, 0);
                        
                        // Color attribute
                        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
                        gl.enableVertexAttribArray(locations.attributes.color);
                        gl.vertexAttribPointer(locations.attributes.color, 3, gl.FLOAT, false, 0, 0);
                    }},
                    
                    // Main rendering function
                    render() {{
                        // Clear canvas with dark background
                        gl.clearColor(0.12, 0.12, 0.15, 1.0);
                        gl.clear(gl.COLOR_BUFFER_BIT);
                        
                        // Use shader program
                        gl.useProgram(program);
                        
                        // Update uniforms (EXACT COPY FROM WORKING EXAMPLE)
                        gl.uniformMatrix3fv(locations.uniforms.rotationMatrix, false, navigation.rotationMatrix);
                        gl.uniform2f(locations.uniforms.translation, ...navigation.translation);
                        gl.uniform1f(locations.uniforms.depth, navigation.depth);
                        
                        // Draw geometry
                        gl.drawArrays(gl.POINTS, 0, pointCount);
                    }}
                }};
                
                // ===== INTERACTION SYSTEM =====
                const InteractionSystem = {{
                    // Constants for interaction
                    DRAG_THRESHOLD: 0.1,
                    DEPTH_LIMITS: {{ min: 0.5, max: 20.0 }},
                    
                    // Mouse button mappings
                    MOUSE_BUTTONS: {{
                        LEFT: 0,
                        MIDDLE: 1,
                        RIGHT: 2
                    }},
                    
                    // Handle rotation input
                    handleRotation(deltaX, deltaY) {{
                        const {{ rotation: sensitivity }} = navigation.sensitivity;
                        let needsRedraw = false;
                        
                        // Yaw rotation (around world Y-axis)
                        if (Math.abs(deltaX) > this.DRAG_THRESHOLD) {{
                            const yRotation = MatrixUtils.createRotation(0, 1, 0, deltaX * sensitivity);
                            navigation.rotationMatrix = MatrixUtils.multiply(navigation.rotationMatrix, yRotation);
                            needsRedraw = true;
                        }}
                        
                        // Pitch rotation (around world X-axis)
                        if (Math.abs(deltaY) > this.DRAG_THRESHOLD) {{
                            const xRotation = MatrixUtils.createRotation(1, 0, 0, deltaY * sensitivity);
                            navigation.rotationMatrix = MatrixUtils.multiply(navigation.rotationMatrix, xRotation);
                            needsRedraw = true;
                        }}
                        
                        return needsRedraw;
                    }},
                    
                    // Handle translation input
                    handleTranslation(deltaX, deltaY) {{
                        const {{ translation: sensitivity }} = navigation.sensitivity;
                        
                        navigation.translation[0] += deltaX * sensitivity;
                        navigation.translation[1] -= deltaY * sensitivity;  // Invert Y for intuitive control
                        
                        return true; // Always redraw for translation
                    }},
                    
                    // Handle depth input
                    handleDepth(wheelDelta) {{
                        const {{ depth: sensitivity }} = navigation.sensitivity;
                        
                        // Scroll up = move forward (decrease depth), scroll down = move backward
                        navigation.depth += (wheelDelta > 0) ? sensitivity : -sensitivity;
                        navigation.depth = Math.max(this.DEPTH_LIMITS.min, Math.min(this.DEPTH_LIMITS.max, navigation.depth));
                        
                        return true; // Always redraw for depth change
                    }}
                }};
                
                // ===== EVENT LISTENERS =====
                canvas.addEventListener('mousedown', (e) => {{
                    e.preventDefault();
                    
                    const {{ MOUSE_BUTTONS }} = InteractionSystem;
                    
                    switch (e.button) {{
                        case MOUSE_BUTTONS.LEFT:
                            mouse.isLeftDragging = true;
                            canvas.style.cursor = 'grabbing';
                            break;
                        case MOUSE_BUTTONS.RIGHT:
                            mouse.isRightDragging = true;
                            canvas.style.cursor = 'move';
                            break;
                    }}
                    
                    mouse.lastX = e.clientX;
                    mouse.lastY = e.clientY;
                }});
                
                canvas.addEventListener('mousemove', (e) => {{
                    const deltaX = e.clientX - mouse.lastX;
                    const deltaY = e.clientY - mouse.lastY;
                    
                    let needsRedraw = false;
                    
                    if (mouse.isLeftDragging) {{
                        needsRedraw = InteractionSystem.handleRotation(deltaX, deltaY);
                    }} else if (mouse.isRightDragging) {{
                        needsRedraw = InteractionSystem.handleTranslation(deltaX, deltaY);
                    }}
                    
                    if (needsRedraw) {{
                        RenderSystem.render();
                    }}
                    
                    mouse.lastX = e.clientX;
                    mouse.lastY = e.clientY;
                }});
                
                canvas.addEventListener('mouseup', () => {{
                    mouse.isLeftDragging = false;
                    mouse.isRightDragging = false;
                    canvas.style.cursor = 'default';
                }});
                
                canvas.addEventListener('contextmenu', (e) => e.preventDefault());
                
                canvas.addEventListener('wheel', (e) => {{
                    e.preventDefault();
                    
                    if (InteractionSystem.handleDepth(e.deltaY)) {{
                        RenderSystem.render();
                    }}
                }});
                
                // ===== INITIALIZATION =====
                try {{
                    // Setup vertex attributes once (performance optimization)
                    RenderSystem.setupVertexAttributes();
                    
                    // Initial render
                    RenderSystem.render();
                    
                    // Return success message with system info
                    const webglInfo = gl.getParameter(gl.VERSION);
                    console.log('WebGL viewer initialized for {component_id}:', pointCount, 'points');
                    return `✓ Professional WebGL Viewer Ready • ${{pointCount}} points • ${{webglInfo}}`;
                    
                }} catch (error) {{
                    console.error('WebGL initialization error for {component_id}:', error);
                    return `✗ Initialization Error: ${{error.message}}`;
                }}
                
            }} catch (error) {{
                console.error('WebGL callback error for {component_id}:', error);
                return 'Error: ' + error.message;
            }}
        }}
        """,
        Output(f'{component_id}-status', 'children'),
        Input(f'{component_id}-trigger', 'children')
    )
