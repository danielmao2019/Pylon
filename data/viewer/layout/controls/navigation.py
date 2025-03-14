"""UI components for dataset navigation."""
from dash import html, dcc


def create_navigation_controls(datapoint_index=0, min_index=0, max_index=10):
    """
    Create navigation controls for browsing through dataset items.
    
    Args:
        datapoint_index: Current index
        min_index: Minimum index value
        max_index: Maximum index value
        
    Returns:
        html.Div containing navigation controls
    """
    # Create marks at appropriate intervals
    marks = {}
    if max_index <= 10:
        # If less than 10 items, mark each one
        marks = {i: str(i) for i in range(min_index, max_index + 1)}
    else:
        # Otherwise, create marks at regular intervals
        step = max(1, (max_index - min_index) // 10)
        marks = {i: str(i) for i in range(min_index, max_index + 1, step)}
        # Always include the last index
        marks[max_index] = str(max_index)
    
    return html.Div([
        html.Div([
            html.Label("Navigate Datapoints:"),
            html.Div([
                dcc.Slider(
                    id='datapoint-index-slider',
                    min=min_index,
                    max=max_index,
                    value=datapoint_index,
                    marks=marks,
                    step=1
                ),
            ], style={'flex': 1, 'margin-right': '20px'}),
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '10px'}),

        html.Div([
            html.Button("⏮ Prev",
                id='prev-btn',
                n_clicks=0,
                style={
                    'background-color': '#e7e7e7',
                    'border': 'none',
                    'padding': '10px 20px',
                    'margin-right': '10px',
                    'border-radius': '5px',
                    'cursor': 'pointer'
                }
            ),
            html.Button("Next ⏭",
                id='next-btn',
                n_clicks=0,
                style={
                    'background-color': '#e7e7e7',
                    'border': 'none',
                    'padding': '10px 20px',
                    'border-radius': '5px',
                    'cursor': 'pointer'
                }
            ),
            html.Div(id='current-index-display',
                children=f"Index: {datapoint_index} / {max_index}",
                style={'display': 'inline-block', 'margin-left': '20px', 'font-weight': 'bold'}
            ),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start'}),
    ], style={'margin-top': '20px', 'padding': '10px 0'})
