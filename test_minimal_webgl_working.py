#!/usr/bin/env python3
"""
Minimal working example of WebGL point cloud visualization with interactive 3D navigation.

This file demonstrates the professional 3D navigation system that was successfully 
implemented and should be kept for future reference.

Key features:
- Left mouse: Orbit camera (yaw/pitch only, no roll)
- Right mouse: Pan in screen plane
- Scroll/Middle mouse: Zoom
- Colored cube with 6 different face colors for testing rotation
"""

import dash
from dash import html, Input, Output, clientside_callback
import numpy as np

# Create colored cube with 6 faces
def create_colored_cube(density=8):
    """Create a cube with 6 different colored faces."""
    points = []
    colors = []
    
    # Define 6 face colors
    face_colors = [
        [1, 0, 0],  # Red - Front face (z=1)
        [0, 1, 0],  # Green - Back face (z=0)  
        [0, 0, 1],  # Blue - Right face (x=1)
        [1, 1, 0],  # Yellow - Left face (x=0)
        [1, 0, 1],  # Magenta - Top face (y=1)
        [0, 1, 1],  # Cyan - Bottom face (y=0)
    ]
    
    # Generate points for each face
    for i in range(density):
        for j in range(density):
            u = i / (density - 1) if density > 1 else 0
            v = j / (density - 1) if density > 1 else 0
            
            # Front face (z=1) - Red
            points.append([u, v, 1])
            colors.append(face_colors[0])
            
            # Back face (z=0) - Green
            points.append([u, v, 0])
            colors.append(face_colors[1])
            
            # Right face (x=1) - Blue
            points.append([1, u, v])
            colors.append(face_colors[2])
            
            # Left face (x=0) - Yellow
            points.append([0, u, v])
            colors.append(face_colors[3])
            
            # Top face (y=1) - Magenta
            points.append([u, 1, v])
            colors.append(face_colors[4])
            
            # Bottom face (y=0) - Cyan
            points.append([u, 0, v])
            colors.append(face_colors[5])
    
    return points, colors

# Create the cube
points, colors = create_colored_cube(density=8)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Minimal WebGL Point Cloud Test", style={'textAlign': 'center'}),
    html.Div([
        html.Canvas(
            id='webgl-canvas',
            width=800,
            height=600,
            style={
                'border': '1px solid black',
                'display': 'block',
                'margin': '0 auto',
                'cursor': 'grab'
            }
        )
    ]),
    html.Div([
        html.P("Controls:", style={'fontWeight': 'bold'}),
        html.P("• Left mouse drag: Orbit camera (yaw/pitch)"),
        html.P("• Right mouse drag: Pan in screen plane"),  
        html.P("• Scroll wheel: Zoom in/out"),
        html.P("• Middle mouse drag: Zoom with vertical movement"),
    ], style={'textAlign': 'center', 'marginTop': '20px'})
])

# Clientside callback that triggers on page load
clientside_callback(
    f"""
    function() {{
        const canvas = document.getElementById('webgl-canvas');
        if (!canvas) return 'Canvas not found';
        
        const gl = canvas.getContext('webgl');
        if (!gl) {{
            return 'WebGL not supported';
        }}
        
        // Vertex shader source
        const vertexShaderSource = `
            attribute vec3 a_position;
            attribute vec3 a_color;
            uniform mat4 u_matrix;
            varying vec3 v_color;
            
            void main() {{
                gl_Position = u_matrix * vec4(a_position, 1.0);
                gl_PointSize = 4.0;
                v_color = a_color;
            }}
        `;
        
        // Fragment shader source
        const fragmentShaderSource = `
            precision mediump float;
            varying vec3 v_color;
            
            void main() {{
                gl_FragColor = vec4(v_color, 1.0);
            }}
        `;
        
        // Shader compilation function
        function createShader(gl, type, source) {{
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
                console.error('Shader compilation error:', gl.getShaderInfoLog(shader));
                gl.deleteShader(shader);
                return null;
            }}
            return shader;
        }}
        
        // Create and link shader program
        const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
        const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
        
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
            console.error('Program linking error:', gl.getProgramInfoLog(program));
            return 'Program linking failed';
        }}
        
        // Get attribute and uniform locations
        const positionLocation = gl.getAttribLocation(program, 'a_position');
        const colorLocation = gl.getAttribLocation(program, 'a_color');
        const matrixLocation = gl.getUniformLocation(program, 'u_matrix');
        
        // Point cloud data
        const points = {points};
        const colors = {colors};
        
        // Create and populate position buffer
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(points.flat()), gl.STATIC_DRAW);
        
        // Create and populate color buffer
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors.flat()), gl.STATIC_DRAW);
        
        // Matrix math utilities
        function multiplyMatrices(a, b) {{
            const result = new Array(16).fill(0);
            for (let i = 0; i < 4; i++) {{
                for (let j = 0; j < 4; j++) {{
                    for (let k = 0; k < 4; k++) {{
                        result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
                    }}
                }}
            }}
            return result;
        }}
        
        function perspectiveMatrix(fov, aspect, near, far) {{
            const f = Math.tan(Math.PI * 0.5 - 0.5 * fov);
            const rangeInv = 1.0 / (near - far);
            return [
                f / aspect, 0, 0, 0,
                0, f, 0, 0,
                0, 0, (near + far) * rangeInv, -1,
                0, 0, near * far * rangeInv * 2, 0
            ];
        }}
        
        function lookAtMatrix(eye, target, up) {{
            const zAxis = [eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]];
            const zLen = Math.sqrt(zAxis[0]*zAxis[0] + zAxis[1]*zAxis[1] + zAxis[2]*zAxis[2]);
            zAxis[0] /= zLen; zAxis[1] /= zLen; zAxis[2] /= zLen;
            
            const xAxis = [
                up[1] * zAxis[2] - up[2] * zAxis[1],
                up[2] * zAxis[0] - up[0] * zAxis[2],
                up[0] * zAxis[1] - up[1] * zAxis[0]
            ];
            const xLen = Math.sqrt(xAxis[0]*xAxis[0] + xAxis[1]*xAxis[1] + xAxis[2]*xAxis[2]);
            xAxis[0] /= xLen; xAxis[1] /= xLen; xAxis[2] /= xLen;
            
            const yAxis = [
                zAxis[1] * xAxis[2] - zAxis[2] * xAxis[1],
                zAxis[2] * xAxis[0] - zAxis[0] * xAxis[2],
                zAxis[0] * xAxis[1] - zAxis[1] * xAxis[0]
            ];
            
            return [
                xAxis[0], yAxis[0], zAxis[0], 0,
                xAxis[1], yAxis[1], zAxis[1], 0,
                xAxis[2], yAxis[2], zAxis[2], 0,
                -(xAxis[0] * eye[0] + xAxis[1] * eye[1] + xAxis[2] * eye[2]),
                -(yAxis[0] * eye[0] + yAxis[1] * eye[1] + yAxis[2] * eye[2]),
                -(zAxis[0] * eye[0] + zAxis[1] * eye[1] + zAxis[2] * eye[2]),
                1
            ];
        }}
        
        // Camera state for 3D navigation
        let camera = {{
            yaw: 0.5,      // Horizontal rotation around target
            pitch: 0.3,    // Vertical rotation around target  
            distance: 3,   // Distance from target
            targetX: 0.5, targetY: 0.5, targetZ: 0.5,  // Look-at point
            zoom: 1
        }};
        
        // Mouse interaction state
        let isMouseDown = false;
        let mouseButton = -1;
        let lastMouseX = 0;
        let lastMouseY = 0;
        
        // Render function
        function render() {{
            // Set viewport
            gl.viewport(0, 0, canvas.width, canvas.height);
            
            // Clear canvas
            gl.clearColor(0.1, 0.1, 0.1, 1.0);  // Dark background
            gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
            gl.enable(gl.DEPTH_TEST);
            
            // Calculate camera position from spherical coordinates
            const camX = camera.targetX + camera.distance * Math.cos(camera.pitch) * Math.cos(camera.yaw);
            const camY = camera.targetY + camera.distance * Math.cos(camera.pitch) * Math.sin(camera.yaw);
            const camZ = camera.targetZ + camera.distance * Math.sin(camera.pitch);
            
            // Create view and projection matrices
            const viewMatrix = lookAtMatrix(
                [camX, camY, camZ],
                [camera.targetX, camera.targetY, camera.targetZ],
                [0, 0, 1]
            );
            
            const aspect = canvas.width / canvas.height;
            const projMatrix = perspectiveMatrix(Math.PI / 4 * camera.zoom, aspect, 0.1, 100);
            const mvpMatrix = multiplyMatrices(projMatrix, viewMatrix);
            
            // Use shader program
            gl.useProgram(program);
            
            // Set up position attribute
            gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
            gl.enableVertexAttribArray(positionLocation);
            gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
            
            // Set up color attribute
            gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
            gl.enableVertexAttribArray(colorLocation);
            gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
            
            // Set transformation matrix
            gl.uniformMatrix4fv(matrixLocation, false, mvpMatrix);
            
            // Draw points
            gl.drawArrays(gl.POINTS, 0, points.length);
        }}
        
        // Mouse event handlers for 3D navigation
        canvas.addEventListener('mousedown', (e) => {{
            isMouseDown = true;
            mouseButton = e.button;
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            canvas.style.cursor = 'grabbing';
            e.preventDefault();
        }});
        
        canvas.addEventListener('mouseup', () => {{
            isMouseDown = false;
            mouseButton = -1;
            canvas.style.cursor = 'grab';
        }});
        
        canvas.addEventListener('mouseleave', () => {{
            isMouseDown = false;
            mouseButton = -1;
            canvas.style.cursor = 'grab';
        }});
        
        canvas.addEventListener('mousemove', (e) => {{
            if (!isMouseDown) return;
            
            const deltaX = e.clientX - lastMouseX;
            const deltaY = e.clientY - lastMouseY;
            const sensitivity = 0.01;
            
            if (mouseButton === 0) {{
                // Left mouse: Orbit camera (yaw/pitch only)
                camera.yaw -= deltaX * sensitivity;
                camera.pitch += deltaY * sensitivity;
                
                // Clamp pitch to avoid gimbal lock
                camera.pitch = Math.max(-Math.PI/2 + 0.1, Math.min(Math.PI/2 - 0.1, camera.pitch));
                
            }} else if (mouseButton === 2) {{
                // Right mouse: Pan in screen plane
                const rightX = -Math.sin(camera.yaw);
                const rightY = Math.cos(camera.yaw);
                const rightZ = 0;
                
                const upX = -Math.cos(camera.yaw) * Math.sin(camera.pitch);
                const upY = -Math.sin(camera.yaw) * Math.sin(camera.pitch);
                const upZ = Math.cos(camera.pitch);
                
                // Apply pan movement
                camera.targetX += rightX * (-deltaX) * sensitivity - upX * (-deltaY) * sensitivity;
                camera.targetY += rightY * (-deltaX) * sensitivity - upY * (-deltaY) * sensitivity;
                camera.targetZ += rightZ * (-deltaX) * sensitivity - upZ * (-deltaY) * sensitivity;
                
            }} else if (mouseButton === 1) {{
                // Middle mouse: Zoom with vertical movement
                camera.zoom *= (1 + (-deltaY) * sensitivity);
                camera.zoom = Math.max(0.1, Math.min(5.0, camera.zoom));
            }}
            
            lastMouseX = e.clientX;
            lastMouseY = e.clientY;
            render();
        }});
        
        // Mouse wheel for zoom
        canvas.addEventListener('wheel', (e) => {{
            const zoomSpeed = 0.1;
            if (e.deltaY > 0) {{
                camera.zoom *= (1 + zoomSpeed);
            }} else {{
                camera.zoom *= (1 - zoomSpeed);
            }}
            camera.zoom = Math.max(0.1, Math.min(5.0, camera.zoom));
            render();
            e.preventDefault();
        }});
        
        // Disable context menu
        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        // Initial render
        render();
        
        return 'WebGL cube rendered with ' + points.length + ' points!';
    }}
    """,
    Output('webgl-canvas', 'title'),  # Use a simple output that doesn't need a trigger
    Input('webgl-canvas', 'id')       # Trigger when canvas component is ready
)

if __name__ == '__main__':
    app.run(debug=True, port=8888)