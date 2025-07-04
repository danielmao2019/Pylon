#!/usr/bin/env python3
"""
Minimal working example of WebGL point cloud visualization.

This should show a colorful cube made of many points (point cloud) that auto-rotates.
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("WebGL Point Cloud Cube", style={'textAlign': 'center'}),
    html.Canvas(
        id='webgl-canvas',
        width=800,
        height=600,
        style={'border': '1px solid black', 'display': 'block', 'margin': '0 auto'}
    ),
    html.Div(id='status', style={'textAlign': 'center', 'marginTop': '10px'}),
])

# Clientside callback that creates and renders a cube point cloud
clientside_callback(
    """
    function() {
        const canvas = document.getElementById('webgl-canvas');
        const gl = canvas.getContext('webgl');
        
        if (!gl) {
            return 'WebGL not supported';
        }
        
        // Simple vertex shader
        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, `
            attribute vec3 position;
            attribute vec3 color;
            uniform mat4 matrix;
            varying vec3 vColor;
            
            void main() {
                gl_Position = matrix * vec4(position, 1.0);
                gl_PointSize = 3.0;
                vColor = color;
            }
        `);
        gl.compileShader(vertexShader);
        
        // Simple fragment shader
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, `
            precision mediump float;
            varying vec3 vColor;
            
            void main() {
                gl_FragColor = vec4(vColor, 1.0);
            }
        `);
        gl.compileShader(fragmentShader);
        
        // Create program
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        
        // Get locations
        const positionLoc = gl.getAttribLocation(program, 'position');
        const colorLoc = gl.getAttribLocation(program, 'color');
        const matrixLoc = gl.getUniformLocation(program, 'matrix');
        
        // Create cube point cloud (6 faces with many points each)
        const positions = [];
        const colors = [];
        const density = 10; // 10x10 points per face
        
        // Face colors
        const faceColors = [
            [1, 0, 0], // Red - Front face (z=1)
            [0, 1, 0], // Green - Back face (z=0)  
            [0, 0, 1], // Blue - Right face (x=1)
            [1, 1, 0], // Yellow - Left face (x=0)
            [1, 0, 1], // Magenta - Top face (y=1)
            [0, 1, 1], // Cyan - Bottom face (y=0)
        ];
        
        // Generate points for each face
        for (let i = 0; i < density; i++) {
            for (let j = 0; j < density; j++) {
                const u = i / (density - 1); // 0 to 1
                const v = j / (density - 1); // 0 to 1
                
                // Front face (z=1) - Red
                positions.push(u, v, 1);
                colors.push(...faceColors[0]);
                
                // Back face (z=0) - Green
                positions.push(u, v, 0);
                colors.push(...faceColors[1]);
                
                // Right face (x=1) - Blue
                positions.push(1, u, v);
                colors.push(...faceColors[2]);
                
                // Left face (x=0) - Yellow
                positions.push(0, u, v);
                colors.push(...faceColors[3]);
                
                // Top face (y=1) - Magenta
                positions.push(u, 1, v);
                colors.push(...faceColors[4]);
                
                // Bottom face (y=0) - Cyan
                positions.push(u, 0, v);
                colors.push(...faceColors[5]);
            }
        }
        
        const numPoints = positions.length / 3;
        
        // Create buffers
        const posBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        
        const colorBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(colors), gl.STATIC_DRAW);
        
        // Rotation angle
        let angle = 0;
        
        function render() {
            // Clear with dark background
            gl.clearColor(0.1, 0.1, 0.1, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            
            // Use program
            gl.useProgram(program);
            
            // Create rotation matrix (rotate around Y axis)
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            const matrix = new Float32Array([
                c, 0, s, 0,
                0, 1, 0, 0,
                -s, 0, c, 0,
                -0.5, -0.5, -2, 1  // Translate to center and move back
            ]);
            
            // Set matrix
            gl.uniformMatrix4fv(matrixLoc, false, matrix);
            
            // Set position attribute
            gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
            gl.enableVertexAttribArray(positionLoc);
            gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
            
            // Set color attribute  
            gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
            gl.enableVertexAttribArray(colorLoc);
            gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 0, 0);
            
            // Draw all points
            gl.drawArrays(gl.POINTS, 0, numPoints);
            
            // Update angle for auto-rotation
            angle += 0.02;
            
            requestAnimationFrame(render);
        }
        
        render();
        
        return 'Cube point cloud with ' + numPoints + ' colored points rotating!';
    }
    """,
    Output('status', 'children'),
    Input('webgl-canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8888)