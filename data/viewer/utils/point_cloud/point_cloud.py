"""WebGL-based point cloud visualization utilities."""
from typing import Dict, Optional, Union, Any
import os
import numpy as np
import torch
import json
import uuid
from dash import html
from data.viewer.utils.segmentation import get_color


def point_cloud_to_numpy(points: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert a PyTorch tensor to a displayable point cloud."""
    if isinstance(points, torch.Tensor):
        return points.cpu().numpy()
    return points




def detect_webgl_support() -> bool:
    """Detect if WebGL is supported in the current environment."""
    # In a real implementation, this would be done client-side
    # For now, assume WebGL is available (most modern browsers support it)
    return True


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


def create_webgl_point_cloud_component(
    points: Union[torch.Tensor, np.ndarray],
    colors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Point Cloud",
    point_size: float = 2,
    point_opacity: float = 0.8,
    camera_state: Optional[Dict[str, Any]] = None
) -> html.Div:
    """Create a WebGL-based point cloud visualization component."""
    
    # Convert inputs to numpy
    points = point_cloud_to_numpy(points)
    if colors is not None:
        colors = point_cloud_to_numpy(colors)
    if labels is not None:
        labels = point_cloud_to_numpy(labels)
    
    # Use all points - no downsampling
    original_count = len(points)
    
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
    
    # Generate unique ID for this component
    component_id = f"webgl-point-cloud-{uuid.uuid4().hex[:8]}"
    
    # Create the WebGL component
    return html.Div([
        html.H4(title, style={'margin-bottom': '10px'}),
        
        # Performance info
        html.Div([
            html.Span(f"Displaying {len(points):,} points"),
            html.Span(f" • Point size: {point_size} • Opacity: {point_opacity}")
        ], style={'font-size': '12px', 'color': '#666', 'margin-bottom': '5px'}),
        
        # WebGL container
        html.Div(
            id=component_id,
            style={
                'width': '100%',
                'height': '500px',
                'border': '1px solid #ddd',
                'border-radius': '4px',
                'position': 'relative',
                'overflow': 'hidden'
            }
        ),
        
        # Load Three.js and WebGL Point Cloud renderer if not already loaded
        html.Script(f"""
        if (typeof THREE === 'undefined') {{
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r158/three.min.js';
            script.onload = function() {{
                // Load OrbitControls
                const controlsScript = document.createElement('script');
                controlsScript.src = 'https://cdn.jsdelivr.net/npm/three@0.158.0/examples/js/controls/OrbitControls.js';
                controlsScript.onload = function() {{
                    // Load our WebGL point cloud renderer
                    const webglScript = document.createElement('script');
                    webglScript.text = `{WEBGL_POINT_CLOUD_JS}`;
                    document.head.appendChild(webglScript);
                }};
                document.head.appendChild(controlsScript);
            }};
            document.head.appendChild(script);
        }} else if (typeof initWebGLPointCloud === 'undefined') {{
            // Three.js loaded but our renderer is not
            const webglScript = document.createElement('script');
            webglScript.text = `{WEBGL_POINT_CLOUD_JS}`;
            document.head.appendChild(webglScript);
        }}
        """),
        
        # WebGL initialization script - delay execution to ensure Three.js is loaded
        html.Script(f"""
        (function() {{
            const containerId = '{component_id}';
            
            function initWhenReady() {{
                const container = document.getElementById(containerId);
                if (!container) return;
                
                // Check if Three.js is loaded
                if (typeof THREE === 'undefined') {{
                    setTimeout(initWhenReady, 100);
                    return;
                }}
                
                // WebGL data
                const pointCloudData = {json.dumps(webgl_data)};
                const pointSize = {point_size};
                const pointOpacity = {point_opacity};
                const cameraState = {json.dumps(camera_state)};
                
                // Initialize WebGL point cloud renderer
                if (typeof initWebGLPointCloud === 'function') {{
                    initWebGLPointCloud(container, pointCloudData, pointSize, pointOpacity, cameraState);
                }} else {{
                    // Fallback implementation
                    container.innerHTML = `
                        <div style="padding: 20px; text-align: center; color: #155724; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 4px;">
                            <h5>🚀 WebGL Point Cloud (Basic)</h5>
                            <p><strong>${{pointCloudData.point_count.toLocaleString()}}</strong> points loaded</p>
                            <p>Bounding box: ${{pointCloudData.bbox_size.toFixed(2)}} units</p>
                            <p>Point size: ${{pointSize}} • Opacity: ${{pointOpacity}}</p>
                            <br>
                            <p><em>Three.js WebGL renderer is initializing...</em></p>
                            <p><small>Please wait for full WebGL support to load</small></p>
                        </div>
                    `;
                }}
            }}
            
            initWhenReady();
        }})();
        """),
        
        # Controls info
        html.Div([
            html.Strong("Controls: "),
            html.Span("Left click + drag: Rotate • Right click + drag: Pan • Scroll: Zoom • R: Reset view • +/-: Point size")
        ], style={
            'font-size': '11px', 
            'color': '#888', 
            'margin-top': '5px',
            'padding': '5px',
            'background-color': '#f8f9fa',
            'border-radius': '3px'
        })
    ])


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
    
    try:
        return create_webgl_point_cloud_component(
            points=points,
            colors=colors,
            labels=labels,
            title=title,
            point_size=point_size,
            point_opacity=point_opacity,
            camera_state=camera_state
        )
    except Exception as e:
        # Fallback to simple HTML display on error
        points_np = point_cloud_to_numpy(points)
        return html.Div([
            html.H4(f"{title} (Fallback Mode)", style={'color': '#d63384'}),
            html.P(f"WebGL rendering failed: {str(e)}"),
            html.P(f"Point cloud statistics: {len(points_np):,} points"),
            html.P("Please check browser WebGL support or contact support.")
        ], style={
            'padding': '20px',
            'border': '1px solid #f5c6cb',
            'background-color': '#f8d7da',
            'border-radius': '4px',
            'color': '#721c24'
        })


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


# WebGL JavaScript code for Three.js point cloud rendering
WEBGL_POINT_CLOUD_JS = """
/**
 * WebGL Point Cloud Renderer using Three.js
 * High-performance point cloud visualization for large datasets
 */

// Global Three.js imports (assumes Three.js is loaded)
const { Scene, PerspectiveCamera, WebGLRenderer, BufferGeometry, BufferAttribute, 
        PointsMaterial, Points, Color, Vector3, Box3 } = THREE;

// OrbitControls for camera interaction
const { OrbitControls } = THREE;

/**
 * Initialize WebGL point cloud renderer with Three.js
 * @param {HTMLElement} container - DOM container element
 * @param {Object} data - Point cloud data object
 * @param {number} pointSize - Size of points
 * @param {number} opacity - Point opacity
 * @param {Object} cameraState - Initial camera state
 */
function initWebGLPointCloud(container, data, pointSize, opacity, cameraState) {
    // Check WebGL support
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    
    if (!gl) {
        // Fallback for no WebGL support
        container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #856404; background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px;">
                <h5>WebGL Not Supported</h5>
                <p>Your browser doesn't support WebGL. Please use a modern browser or enable WebGL.</p>
                <p>Point cloud data: ${data.point_count.toLocaleString()} points</p>
            </div>
        `;
        return;
    }

    try {
        // Create Three.js scene
        const scene = new Scene();
        scene.background = new Color(0xffffff);

        // Set up camera
        const aspect = container.clientWidth / container.clientHeight;
        const camera = new PerspectiveCamera(75, aspect, 0.1, data.bbox_size * 10);
        
        // Position camera based on provided state or defaults
        const cameraPos = cameraState.position || [data.bbox_size * 2, data.bbox_size * 2, data.bbox_size * 2];
        camera.position.set(cameraPos[0], cameraPos[1], cameraPos[2]);
        
        // Set camera target
        const target = cameraState.target || data.bbox_center;
        camera.lookAt(new Vector3(target[0], target[1], target[2]));

        // Create WebGL renderer
        const renderer = new WebGLRenderer({ 
            antialias: true,
            alpha: true,
            preserveDrawingBuffer: true
        });
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2)); // Limit for performance
        
        // Clear container and add canvas
        container.innerHTML = '';
        container.appendChild(renderer.domElement);

        // Create point cloud geometry
        const geometry = new BufferGeometry();
        
        // Set positions (convert flat array to Float32Array)
        const positions = new Float32Array(data.positions);
        geometry.setAttribute('position', new BufferAttribute(positions, 3));
        
        // Set colors (convert flat array to Float32Array)
        const colors = new Float32Array(data.colors);
        geometry.setAttribute('color', new BufferAttribute(colors, 3));

        // Create material
        const material = new PointsMaterial({
            size: pointSize,
            opacity: opacity,
            transparent: opacity < 1.0,
            vertexColors: true,
            sizeAttenuation: true
        });

        // Create points object
        const points = new Points(geometry, material);
        scene.add(points);

        // Set up orbit controls for interaction
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set(target[0], target[1], target[2]);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.enablePan = true;
        controls.enableZoom = true;
        controls.enableRotate = true;
        
        // Control limits
        controls.minDistance = data.bbox_size * 0.1;
        controls.maxDistance = data.bbox_size * 5;
        
        // Performance optimizations
        controls.enableKeys = false; // Disable keyboard controls for performance
        controls.screenSpacePanning = false;

        // Animation loop
        let animationId;
        function animate() {
            animationId = requestAnimationFrame(animate);
            
            // Update controls
            controls.update();
            
            // Render scene
            renderer.render(scene, camera);
        }

        // Handle window resize
        const resizeObserver = new ResizeObserver(entries => {
            const entry = entries[0];
            const { width, height } = entry.contentRect;
            
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        });
        resizeObserver.observe(container);

        // Keyboard shortcuts
        const keyHandler = (event) => {
            switch(event.key.toLowerCase()) {
                case 'r':
                    // Reset view
                    camera.position.set(cameraPos[0], cameraPos[1], cameraPos[2]);
                    controls.target.set(target[0], target[1], target[2]);
                    controls.update();
                    break;
                case '+':
                case '=':
                    // Increase point size
                    material.size = Math.min(material.size + 0.5, 20);
                    break;
                case '-':
                    // Decrease point size
                    material.size = Math.max(material.size - 0.5, 0.5);
                    break;
            }
        };
        
        // Add event listeners
        document.addEventListener('keydown', keyHandler);

        // Start animation
        animate();

        // Performance monitoring
        let frameCount = 0;
        let lastTime = performance.now();
        const fpsDisplay = document.createElement('div');
        fpsDisplay.style.cssText = `
            position: absolute;
            top: 5px;
            right: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 5px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 12px;
            z-index: 1000;
        `;
        container.appendChild(fpsDisplay);
        container.style.position = 'relative';

        function updateFPS() {
            frameCount++;
            const currentTime = performance.now();
            if (currentTime - lastTime >= 1000) {
                const fps = Math.round((frameCount * 1000) / (currentTime - lastTime));
                fpsDisplay.textContent = `${fps} FPS | ${data.point_count.toLocaleString()} pts`;
                frameCount = 0;
                lastTime = currentTime;
            }
            requestAnimationFrame(updateFPS);
        }
        updateFPS();

        // Cleanup function for component unmounting
        const cleanup = () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
            document.removeEventListener('keydown', keyHandler);
            resizeObserver.disconnect();
            
            // Dispose Three.js resources
            geometry.dispose();
            material.dispose();
            renderer.dispose();
            
            // Clear container
            container.innerHTML = '';
        };

        // Store cleanup function on container for external access
        container._webglCleanup = cleanup;

    } catch (error) {
        // Error fallback
        container.innerHTML = `
            <div style="padding: 20px; text-align: center; color: #721c24; background-color: #f8d7da; border: 1px solid #f5c6cb; border-radius: 4px;">
                <h5>WebGL Initialization Error</h5>
                <p>Error: ${error.message}</p>
                <p>Point count: ${data.point_count.toLocaleString()}</p>
                <details style="margin-top: 10px;">
                    <summary>Error Details</summary>
                    <pre style="text-align: left; font-size: 11px; margin-top: 5px;">${error.stack}</pre>
                </details>
            </div>
        `;
        console.error('WebGL Point Cloud Error:', error);
    }
}

// Make functions available globally
if (typeof window !== 'undefined') {
    window.initWebGLPointCloud = initWebGLPointCloud;
}
"""


def get_webgl_assets() -> str:
    """Get the WebGL JavaScript code for point cloud rendering."""
    return WEBGL_POINT_CLOUD_JS