#!/usr/bin/env python3
"""
Baby step 4: WebGL shaders + multiple points from array
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("WebGL Multiple Points Test"),
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
        
        // Vertex shader with position attribute
        const vertexShaderSource = `
            attribute vec2 aPosition;
            void main() {
                gl_Position = vec4(aPosition, 0.0, 1.0);
                gl_PointSize = 15.0;
            }
        `;
        
        // Fragment shader
        const fragmentShaderSource = `
            precision mediump float;
            void main() {
                gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
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
        
        // Create buffer
        const positionBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.STATIC_DRAW);
        
        // Get attribute location
        const positionLocation = gl.getAttribLocation(program, 'aPosition');
        
        // Set up attribute
        gl.enableVertexAttribArray(positionLocation);
        gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);
        
        // Clear and draw
        gl.clearColor(0.2, 0.2, 0.2, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.drawArrays(gl.POINTS, 0, 4);  // Draw 4 points
        
        return 'Should see 4 red points in square pattern';
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8889)
