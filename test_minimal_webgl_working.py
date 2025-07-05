#!/usr/bin/env python3
"""
Baby step 5: WebGL shaders + multiple colored points
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("WebGL 3D Cube with Yaw/Pitch Rotation"),
    html.Canvas(id='canvas', width=400, height=400, style={'border': '1px solid black'}),
    html.Div(id='output'),
    html.P("Left drag: Rotate • Right drag: Pan • No roll behavior")
])

clientside_callback(
    """
    function() {
        const canvas = document.getElementById('canvas');
        const gl = canvas.getContext('webgl');
        
        if (!gl) {
            return 'WebGL not supported';
        }
        
        // Vertex shader with rotation matrix and translation
        const vertexShaderSource = `
            attribute vec3 aPosition;
            attribute vec3 aColor;
            uniform mat3 uRotationMatrix;
            uniform vec2 uTranslation;
            varying vec3 vColor;
            void main() {
                // Apply rotation matrix
                vec3 rotated = uRotationMatrix * aPosition;
                
                // Apply translation in screen space
                rotated.x += uTranslation.x;
                rotated.y += uTranslation.y;
                
                // Simple 3D to 2D projection
                float z = rotated.z + 3.0;
                gl_Position = vec4(rotated.x / z, rotated.y / z, 0.0, 1.0);
                gl_PointSize = 8.0;
                vColor = aColor;
            }
        `;
        
        // Fragment shader with color varying
        const fragmentShaderSource = `
            precision mediump float;
            varying vec3 vColor;
            void main() {
                gl_FragColor = vec4(vColor, 1.0);
            }
        `;
        
        // Create shaders
        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexShaderSource);
        gl.compileShader(vertexShader);
        
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentShaderSource);
        gl.compileShader(fragmentShader);
        
        // Create program
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        // Create cube faces with moderate density sampling
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
        
        // Generate points for each face
        for (let i = 0; i < density; i++) {
            for (let j = 0; j < density; j++) {
                const u = (i / (density - 1)) - 0.5; // -0.5 to 0.5
                const v = (j / (density - 1)) - 0.5; // -0.5 to 0.5
                
                // Front face (z=0.5) - Red
                positions.push(u, v, 0.5);
                colors.push(...faceColors[0]);
                
                // Back face (z=-0.5) - Green
                positions.push(u, v, -0.5);
                colors.push(...faceColors[1]);
                
                // Right face (x=0.5) - Blue
                positions.push(0.5, u, v);
                colors.push(...faceColors[2]);
                
                // Left face (x=-0.5) - Yellow
                positions.push(-0.5, u, v);
                colors.push(...faceColors[3]);
                
                // Top face (y=0.5) - Magenta
                positions.push(u, 0.5, v);
                colors.push(...faceColors[4]);
                
                // Bottom face (y=-0.5) - Cyan
                positions.push(u, -0.5, v);
                colors.push(...faceColors[5]);
            }
        }
        
        const pointCount = positions.length / 3;
        
        // Create position buffer
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        
        // Create color buffer
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        
        // Get attribute and uniform locations
        const positionLocation = gl.getAttribLocation(program, 'aPosition');
        const colorLocation = gl.getAttribLocation(program, 'aColor');
        const rotationMatrixLocation = gl.getUniformLocation(program, 'uRotationMatrix');
        const translationLocation = gl.getUniformLocation(program, 'uTranslation');
        
        // Rotation and translation state
        let rotationMatrix = [
            1, 0, 0,
            0, 1, 0,
            0, 0, 1
        ];
        let translation = [0, 0];  // Screen space translation [x, y]
        
        let mouseState = {
            isLeftDragging: false,
            isRightDragging: false,
            lastX: 0,
            lastY: 0
        };
        
        // Helper function to create rotation matrix around axis
        function createRotationMatrix(axisX, axisY, axisZ, angle) {
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            const t = 1 - c;
            
            return [
                t*axisX*axisX + c,       t*axisX*axisY - s*axisZ, t*axisX*axisZ + s*axisY,
                t*axisX*axisY + s*axisZ, t*axisY*axisY + c,       t*axisY*axisZ - s*axisX,
                t*axisX*axisZ - s*axisY, t*axisY*axisZ + s*axisX, t*axisZ*axisZ + c
            ];
        }
        
        // Helper function to multiply 3x3 matrices
        function multiplyMatrix3(a, b) {
            return [
                a[0]*b[0] + a[1]*b[3] + a[2]*b[6], a[0]*b[1] + a[1]*b[4] + a[2]*b[7], a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
                a[3]*b[0] + a[4]*b[3] + a[5]*b[6], a[3]*b[1] + a[4]*b[4] + a[5]*b[7], a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
                a[6]*b[0] + a[7]*b[3] + a[8]*b[6], a[6]*b[1] + a[7]*b[4] + a[8]*b[7], a[6]*b[2] + a[7]*b[5] + a[8]*b[8]
            ];
        }
        
        // Set up position attribute
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
        
        // Set up color attribute
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
        
        // Mouse event handlers for rotation and translation
        canvas.addEventListener('mousedown', (e) => {
            e.preventDefault();
            
            if (e.button === 0) {  // Left mouse button - rotation
                mouseState.isLeftDragging = true;
                canvas.style.cursor = 'grabbing';
            } else if (e.button === 2) {  // Right mouse button - translation
                mouseState.isRightDragging = true;
                canvas.style.cursor = 'move';
            }
            
            mouseState.lastX = e.clientX;
            mouseState.lastY = e.clientY;
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const deltaX = e.clientX - mouseState.lastX;
            const deltaY = e.clientY - mouseState.lastY;
            
            if (mouseState.isLeftDragging) {
                // Left mouse: Rotation (yaw and pitch)
                const rotationSensitivity = 0.01;
                
                // Create rotation around Y axis (horizontal mouse movement = yaw)
                if (Math.abs(deltaX) > 0.1) {
                    const yRotation = createRotationMatrix(0, 1, 0, deltaX * rotationSensitivity);
                    rotationMatrix = multiplyMatrix3(rotationMatrix, yRotation);
                }
                
                // Create rotation around X axis (vertical mouse movement = pitch)
                if (Math.abs(deltaY) > 0.1) {
                    const xRotation = createRotationMatrix(1, 0, 0, deltaY * rotationSensitivity);
                    rotationMatrix = multiplyMatrix3(rotationMatrix, xRotation);
                }
                
                drawScene();
            } else if (mouseState.isRightDragging) {
                // Right mouse: Translation in screen plane
                const translationSensitivity = 0.002;
                
                translation[0] += deltaX * translationSensitivity;
                translation[1] -= deltaY * translationSensitivity;  // Invert Y for intuitive control
                
                drawScene();
            }
            
            mouseState.lastX = e.clientX;
            mouseState.lastY = e.clientY;
        });
        
        canvas.addEventListener('mouseup', () => {
            mouseState.isLeftDragging = false;
            mouseState.isRightDragging = false;
            canvas.style.cursor = 'default';
        });
        
        // Disable context menu on right click
        canvas.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });
        
        function drawScene() {
            // Set up attributes
            gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
            gl.enableVertexAttribArray(positionLocation);
            gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
            
            gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
            gl.enableVertexAttribArray(colorLocation);
            gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
            
            // Clear and draw
            gl.clearColor(0.2, 0.2, 0.2, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.useProgram(program);
            
            // Set rotation matrix and translation uniforms
            gl.uniformMatrix3fv(rotationMatrixLocation, false, rotationMatrix);
            gl.uniform2f(translationLocation, translation[0], translation[1]);
            
            gl.drawArrays(gl.POINTS, 0, pointCount);
        }
        
        // Initial draw
        drawScene();
        
        return 'Interactive cube with ' + pointCount + ' colored face points - drag to rotate!';
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8889)
