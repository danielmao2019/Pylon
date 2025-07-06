#!/usr/bin/env python3
"""
Professional 3D WebGL Point Cloud Viewer
Implements rotation (yaw/pitch), panning, and dolly controls with world-space matrix operations.
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Professional 3D Point Cloud Viewer"),
    html.Canvas(
        id='canvas', 
        width=600, 
        height=600, 
        style={
            'border': '2px solid #333',
            'borderRadius': '4px',
            'display': 'block',
            'margin': '20px auto'
        }
    ),
    html.Div(id='output', style={'textAlign': 'center', 'marginTop': '10px'}),
    html.Div([
        html.Strong("Controls: "),
        html.Span("Left drag: Rotate • Right drag: Pan • Scroll: Forward/Backward")
    ], style={
        'textAlign': 'center',
        'fontSize': '14px',
        'color': '#666',
        'marginTop': '10px',
        'padding': '10px',
        'backgroundColor': '#f8f9fa',
        'borderRadius': '4px'
    })
], style={'maxWidth': '800px', 'margin': '0 auto', 'padding': '20px'})

clientside_callback(
    """
    function() {
        // ===== INITIALIZATION =====
        const canvas = document.getElementById('canvas');
        const gl = canvas.getContext('webgl');
        
        if (!gl) {
            return 'Error: WebGL not supported in this browser';
        }
        
        // ===== SHADER SOURCES =====
        const vertexShaderSource = `
            attribute vec3 aPosition;
            attribute vec3 aColor;
            uniform mat3 uRotationMatrix;
            uniform vec2 uTranslation;
            uniform float uDepth;
            varying vec3 vColor;
            
            void main() {
                // Apply world-space rotation matrix
                vec3 rotated = uRotationMatrix * aPosition;
                
                // Apply screen-space translation
                rotated.x += uTranslation.x;
                rotated.y += uTranslation.y;
                
                // Perspective projection with adjustable camera distance
                float z = rotated.z + uDepth;
                gl_Position = vec4(rotated.x / z, rotated.y / z, 0.0, 1.0);
                gl_PointSize = 8.0;
                vColor = aColor;
            }
        `;
        
        const fragmentShaderSource = `
            precision mediump float;
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0);
            }
        `;
        
        // ===== SHADER COMPILATION =====
        function createShader(type, source) {
            const shader = gl.createShader(type);
            gl.shaderSource(shader, source);
            gl.compileShader(shader);
            
            if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
                const error = gl.getShaderInfoLog(shader);
                gl.deleteShader(shader);
                throw new Error('Shader compilation error: ' + error);
            }
            
            return shader;
        }
        
        let program;
        try {
            const vertexShader = createShader(gl.VERTEX_SHADER, vertexShaderSource);
            const fragmentShader = createShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
            
            // Create and link program
            program = gl.createProgram();
            gl.attachShader(program, vertexShader);
            gl.attachShader(program, fragmentShader);
            gl.linkProgram(program);
            
            if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
                throw new Error('Program linking error: ' + gl.getProgramInfoLog(program));
            }
            
            // Clean up shaders
            gl.deleteShader(vertexShader);
            gl.deleteShader(fragmentShader);
        } catch (error) {
            return 'Shader Error: ' + error.message;
        }
        
        // ===== GEOMETRY GENERATION =====
        function generateCubePoints() {
            const density = 6; // Points per edge
            const positions = [];
            const colors = [];
            
            // Define 6 face colors
            const faceColors = [
                [1.0, 0.0, 0.0],  // Red - Front face (z=0.5)
                [0.0, 1.0, 0.0],  // Green - Back face (z=-0.5)
                [0.0, 0.0, 1.0],  // Blue - Right face (x=0.5)
                [1.0, 1.0, 0.0],  // Yellow - Left face (x=-0.5)
                [1.0, 0.0, 1.0],  // Magenta - Top face (y=0.5)
                [0.0, 1.0, 1.0]   // Cyan - Bottom face (y=-0.5)
            ];
            
            // Helper function to add face points
            function addFace(faceIndex, getPosition) {
                for (let i = 0; i < density; i++) {
                    for (let j = 0; j < density; j++) {
                        const u = (i / (density - 1)) - 0.5; // -0.5 to 0.5
                        const v = (j / (density - 1)) - 0.5; // -0.5 to 0.5
                        
                        const pos = getPosition(u, v);
                        positions.push(...pos);
                        colors.push(...faceColors[faceIndex]);
                    }
                }
            }
            
            // Generate all 6 faces
            addFace(0, (u, v) => [u, v, 0.5]);    // Front (Red)
            addFace(1, (u, v) => [u, v, -0.5]);   // Back (Green)
            addFace(2, (u, v) => [0.5, u, v]);    // Right (Blue)
            addFace(3, (u, v) => [-0.5, u, v]);   // Left (Yellow)
            addFace(4, (u, v) => [u, 0.5, v]);    // Top (Magenta)
            addFace(5, (u, v) => [u, -0.5, v]);   // Bottom (Cyan)
            
            return { positions, colors, pointCount: positions.length / 3 };
        }
        
        const { positions, colors, pointCount } = generateCubePoints();
        
        // ===== BUFFER CREATION =====
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        
        // ===== SHADER LOCATIONS =====
        const locations = {
            attributes: {
                position: gl.getAttribLocation(program, 'aPosition'),
                color: gl.getAttribLocation(program, 'aColor')
            },
            uniforms: {
                rotationMatrix: gl.getUniformLocation(program, 'uRotationMatrix'),
                translation: gl.getUniformLocation(program, 'uTranslation'),
                depth: gl.getUniformLocation(program, 'uDepth')
            }
        };
        
        // ===== NAVIGATION STATE =====
        const navigation = {
            // World-space rotation matrix (identity initially)
            rotationMatrix: [1, 0, 0, 0, 1, 0, 0, 0, 1],
            
            // Screen-space translation
            translation: [0, 0],
            
            // Camera distance (lower = closer)
            depth: 3.0,
            
            // Sensitivity settings
            sensitivity: {
                rotation: 0.01,
                translation: 0.002,
                depth: 0.1
            }
        };
        
        // Mouse interaction state
        const mouse = {
            isLeftDragging: false,
            isRightDragging: false,
            lastX: 0,
            lastY: 0
        };
        
        // ===== MATRIX UTILITIES =====
        const MatrixUtils = {
            // Create rotation matrix around arbitrary axis using Rodrigues' formula
            createRotation(axisX, axisY, axisZ, angle) {
                const c = Math.cos(angle);
                const s = Math.sin(angle);
                const t = 1 - c;
                
                return [
                    t*axisX*axisX + c,       t*axisX*axisY - s*axisZ, t*axisX*axisZ + s*axisY,
                    t*axisX*axisY + s*axisZ, t*axisY*axisY + c,       t*axisY*axisZ - s*axisX,
                    t*axisX*axisZ - s*axisY, t*axisY*axisZ + s*axisX, t*axisZ*axisZ + c
                ];
            },
            
            // Multiply two 3x3 matrices: result = a * b
            multiply(a, b) {
                return [
                    a[0]*b[0] + a[1]*b[3] + a[2]*b[6], a[0]*b[1] + a[1]*b[4] + a[2]*b[7], a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
                    a[3]*b[0] + a[4]*b[3] + a[5]*b[6], a[3]*b[1] + a[4]*b[4] + a[5]*b[7], a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
                    a[6]*b[0] + a[7]*b[3] + a[8]*b[6], a[6]*b[1] + a[7]*b[4] + a[8]*b[7], a[6]*b[2] + a[7]*b[5] + a[8]*b[8]
                ];
            }
        };
        
        // ===== RENDERING SYSTEM =====
        const RenderSystem = {
            // One-time setup of vertex attributes (performance optimization)
            setupVertexAttributes() {
                // Position attribute
                gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
                gl.enableVertexAttribArray(locations.attributes.position);
                gl.vertexAttribPointer(locations.attributes.position, 3, gl.FLOAT, false, 0, 0);
                
                // Color attribute
                gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
                gl.enableVertexAttribArray(locations.attributes.color);
                gl.vertexAttribPointer(locations.attributes.color, 3, gl.FLOAT, false, 0, 0);
            },
            
            // Main rendering function
            render() {
                // Clear canvas with dark background
                gl.clearColor(0.12, 0.12, 0.15, 1.0);
                gl.clear(gl.COLOR_BUFFER_BIT);
                
                // Use shader program
                gl.useProgram(program);
                
                // Update uniforms
                gl.uniformMatrix3fv(locations.uniforms.rotationMatrix, false, navigation.rotationMatrix);
                gl.uniform2f(locations.uniforms.translation, ...navigation.translation);
                gl.uniform1f(locations.uniforms.depth, navigation.depth);
                
                // Draw geometry
                gl.drawArrays(gl.POINTS, 0, pointCount);
            }
        };
        
        // ===== INTERACTION SYSTEM =====
        const InteractionSystem = {
            // Constants for interaction
            DRAG_THRESHOLD: 0.1,
            DEPTH_LIMITS: { min: 0.5, max: 10.0 },
            
            // Mouse button mappings
            MOUSE_BUTTONS: {
                LEFT: 0,
                MIDDLE: 1,
                RIGHT: 2
            },
            
            // Handle rotation input
            handleRotation(deltaX, deltaY) {
                const { rotation: sensitivity } = navigation.sensitivity;
                let needsRedraw = false;
                
                // Yaw rotation (around world Y-axis)
                if (Math.abs(deltaX) > this.DRAG_THRESHOLD) {
                    const yRotation = MatrixUtils.createRotation(0, 1, 0, deltaX * sensitivity);
                    navigation.rotationMatrix = MatrixUtils.multiply(navigation.rotationMatrix, yRotation);
                    needsRedraw = true;
                }
                
                // Pitch rotation (around world X-axis)
                if (Math.abs(deltaY) > this.DRAG_THRESHOLD) {
                    const xRotation = MatrixUtils.createRotation(1, 0, 0, deltaY * sensitivity);
                    navigation.rotationMatrix = MatrixUtils.multiply(navigation.rotationMatrix, xRotation);
                    needsRedraw = true;
                }
                
                return needsRedraw;
            },
            
            // Handle translation input
            handleTranslation(deltaX, deltaY) {
                const { translation: sensitivity } = navigation.sensitivity;
                
                navigation.translation[0] += deltaX * sensitivity;
                navigation.translation[1] -= deltaY * sensitivity;  // Invert Y for intuitive control
                
                return true; // Always redraw for translation
            },
            
            // Handle depth input
            handleDepth(wheelDelta) {
                const { depth: sensitivity } = navigation.sensitivity;
                
                // Scroll up = move forward (decrease depth), scroll down = move backward
                navigation.depth += (wheelDelta > 0) ? sensitivity : -sensitivity;
                navigation.depth = Math.max(this.DEPTH_LIMITS.min, Math.min(this.DEPTH_LIMITS.max, navigation.depth));
                
                return true; // Always redraw for depth change
            }
        };
        
        // ===== EVENT LISTENERS =====
        canvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            
            const { MOUSE_BUTTONS } = InteractionSystem;
            
            switch (e.button) {
                case MOUSE_BUTTONS.LEFT:
                    mouse.isLeftDragging = true;
                    canvas.style.cursor = 'grabbing';
                    break;
                case MOUSE_BUTTONS.RIGHT:
                    mouse.isRightDragging = true;
                    canvas.style.cursor = 'move';
                    break;
            }
            
            mouse.lastX = e.clientX;
            mouse.lastY = e.clientY;
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const deltaX = e.clientX - mouse.lastX;
            const deltaY = e.clientY - mouse.lastY;
            
            let needsRedraw = false;
            
            if (mouse.isLeftDragging) {
                needsRedraw = InteractionSystem.handleRotation(deltaX, deltaY);
            } else if (mouse.isRightDragging) {
                needsRedraw = InteractionSystem.handleTranslation(deltaX, deltaY);
            }
            
            if (needsRedraw) {
                RenderSystem.render();
            }
            
            mouse.lastX = e.clientX;
            mouse.lastY = e.clientY;
        });
        
        canvas.addEventListener('mouseup', () => {
            mouse.isLeftDragging = false;
            mouse.isRightDragging = false;
            canvas.style.cursor = 'default';
        });
        
        canvas.addEventListener('contextmenu', (e) => e.preventDefault());
        
        canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            if (InteractionSystem.handleDepth(e.deltaY)) {
                RenderSystem.render();
            }
        });
        
        // ===== INITIALIZATION =====
        try {
            // Setup vertex attributes once (performance optimization)
            RenderSystem.setupVertexAttributes();
            
            // Initial render
            RenderSystem.render();
            
            // Return success message with system info
            const webglInfo = gl.getParameter(gl.VERSION);
            return `✓ Professional 3D Viewer Ready • ${pointCount} points • ${webglInfo}`;
            
        } catch (error) {
            return `✗ Initialization Error: ${error.message}`;
        }
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8889)
