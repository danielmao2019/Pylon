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
        
        // Vertex shader with position and color attributes
        const vertexShaderSource = `
            attribute vec2 aPosition;
            attribute vec3 aColor;
            varying vec3 vColor;
            void main() {
                gl_Position = vec4(aPosition, 0.0, 1.0);
                gl_PointSize = 20.0;
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
        
        // Define 4 points in a square pattern
        const positions = [
            -0.5, -0.5,  // Bottom left
             0.5, -0.5,  // Bottom right
             0.5,  0.5,  // Top right
            -0.5,  0.5   // Top left
        ];
        
        // Define colors for each point (RGB)
        const colors = [
            1.0, 0.0, 0.0,  // Red - Bottom left
            0.0, 1.0, 0.0,  // Green - Bottom right
            0.0, 0.0, 1.0,  // Blue - Top right
            1.0, 1.0, 0.0   // Yellow - Top left
        ];
        
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
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
        
        // Set up color attribute
        gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
        gl.enableVertexAttribArray(colorLocation);
        gl.vertexAttribPointer(colorLocation, 3, gl.FLOAT, false, 0, 0);
        
        // Clear and draw
        gl.clearColor(0.2, 0.2, 0.2, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.drawArrays(gl.POINTS, 0, 4);  // Draw 4 points
        
        return 'Should see 4 colored points: red, green, blue, yellow';
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8889)
