#!/usr/bin/env python3
"""
Baby step 5: WebGL shaders + multiple colored points
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("WebGL Colored Points Test"),
    html.Canvas(id='canvas', width=400, height=400, style={'border': '1px solid black'}),
    html.Div(id='output')
])

clientside_callback(
    """
    function() {
        const canvas = document.getElementById('canvas');
        const gl = canvas.getContext('webgl');
        
        if (!gl) {
            return 'WebGL not supported';
        }
        
        // Vertex shader with 3D position and color attributes
        const vertexShaderSource = `
            attribute vec3 aPosition;
            attribute vec3 aColor;
            varying vec3 vColor;
            void main() {
                // Simple 3D to 2D projection (no rotation yet)
                float z = aPosition.z + 3.0;
                gl_Position = vec4(aPosition.x / z, aPosition.y / z, 0.0, 1.0);
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
        
        // Get attribute locations
        const positionLocation = gl.getAttribLocation(program, 'aPosition');
        const colorLocation = gl.getAttribLocation(program, 'aColor');
        
        // Set up position attribute
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 3, gl.FLOAT, false, 0, 0);
        
        // Set up color attribute
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
        
        // Clear and draw
        gl.clearColor(0.2, 0.2, 0.2, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.drawArrays(gl.POINTS, 0, pointCount);  // Draw all cube face points
        
        return 'Should see cube with ' + pointCount + ' colored face points';
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8889)
