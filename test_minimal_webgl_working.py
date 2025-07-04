#!/usr/bin/env python3
"""
Minimal working example of WebGL point cloud visualization.

This should show a colorful cube made of points that you can rotate with the mouse.
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

# Simple clientside callback that renders points
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
                gl_PointSize = 5.0;
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
        
        // Create cube points (8 corners)
        const positions = [
            // Bottom face
            -1, -1, -1,  // 0
             1, -1, -1,  // 1
             1,  1, -1,  // 2
            -1,  1, -1,  // 3
            // Top face
            -1, -1,  1,  // 4
             1, -1,  1,  // 5
             1,  1,  1,  // 6
            -1,  1,  1,  // 7
        ];
        
        const colors = [
            1, 0, 0,  // Red
            0, 1, 0,  // Green
            0, 0, 1,  // Blue
            1, 1, 0,  // Yellow
            1, 0, 1,  // Magenta
            0, 1, 1,  // Cyan
            1, 1, 1,  // White
            0, 0, 0,  // Black
        ];
        
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
            // Clear
            gl.clearColor(0.2, 0.2, 0.2, 1.0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            
            // Use program
            gl.useProgram(program);
            
            // Create rotation matrix
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            const matrix = new Float32Array([
                c, 0, s, 0,
                0, 1, 0, 0,
                -s, 0, c, 0,
                0, 0, 0, 1
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
            
            // Draw points
            gl.drawArrays(gl.POINTS, 0, 8);
            
            // Update angle
            angle += 0.01;
            
            requestAnimationFrame(render);
        }
        
        render();
        
        return '8 colorful points should be rotating!';
    }
    """,
    Output('status', 'children'),
    Input('webgl-canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8888)