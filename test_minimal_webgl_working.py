#!/usr/bin/env python3
"""
ABSOLUTE MINIMAL WebGL test - just show some points!
"""

import dash
from dash import html, Input, Output, clientside_callback

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("WebGL Test"),
    html.Canvas(id='canvas', width=400, height=400, style={'border': '1px solid black'}),
    html.Div(id='output')
])

clientside_callback(
    """
    function() {
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        // Just draw some colored circles to test if anything works
        ctx.fillStyle = 'red';
        ctx.fillRect(50, 50, 50, 50);
        
        ctx.fillStyle = 'blue';
        ctx.fillRect(150, 150, 50, 50);
        
        ctx.fillStyle = 'green';
        ctx.fillRect(250, 250, 50, 50);
        
        return 'Canvas 2D test - should see 3 colored squares';
    }
    """,
    Output('output', 'children'),
    Input('canvas', 'id')
)

if __name__ == '__main__':
    app.run(debug=True, port=8888)