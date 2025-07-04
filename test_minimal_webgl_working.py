#!/usr/bin/env python3
"""
Baby step 3: WebGL shaders + single red point
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("WebGL Single Red Point Test"),
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
        
        // Simple vertex shader
        const vertexShaderSource = `
            void main() {
                gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
                gl_PointSize = 20.0;
            }
        `;
        
        // Simple fragment shader
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
        
        // Clear and draw
        gl.clearColor(0.2, 0.2, 0.2, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(program);
        gl.drawArrays(gl.POINTS, 0, 1);
        
        return 'Should see one big red point in center of grey canvas';
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8888)