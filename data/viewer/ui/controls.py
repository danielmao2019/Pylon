"""UI control components for the dataset viewer."""
from dash import dcc, html


def create_dataset_selector(available_datasets):
    """
    Create a dataset selector dropdown.
    
    Args:
        available_datasets: Dictionary of available datasets
        
    Returns:
        html.Div containing the dataset selector
    """
    return html.Div([
        html.Label("Select Dataset:"),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[{'label': name, 'value': name} for name in sorted(available_datasets.keys())],
            value=None,
            style={'width': '100%'}
        )
    ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'})


def create_reload_button():
    """
    Create a button to reload datasets.
    
    Returns:
        html.Div containing the reload button
    """
    return html.Div([
        html.Button(
            "Reload Datasets",
            id='reload-button',
            style={
                'background-color': '#007bff',
                'color': 'white',
                'border': 'none',
                'padding': '10px 15px',
                'cursor': 'pointer',
                'border-radius': '5px',
                'margin-top': '20px'
            }
        ),
    ], style={'width': '30%', 'display': 'inline-block', 'text-align': 'right'})


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


def create_3d_controls(visible=False, point_size=2, point_opacity=0.8):
    """
    Create 3D visualization controls.
    
    Args:
        visible: Whether the controls should be visible
        point_size: Initial point size
        point_opacity: Initial point opacity
        
    Returns:
        html.Div containing 3D controls
    """
    style = {'display': 'block' if visible else 'none', 'margin-top': '20px'}
    
    return html.Div([
        html.H3("3D View Controls", style={'margin-top': '0'}),

        html.Label("Point Size"),
        dcc.Slider(
            id='point-size-slider',
            min=1,
            max=10,
            value=point_size,
            marks={i: str(i) for i in [1, 3, 5, 7, 10]},
            step=0.5
        ),

        html.Label("Point Opacity", style={'margin-top': '20px'}),
        dcc.Slider(
            id='point-opacity-slider',
            min=0.1,
            max=1.0,
            value=point_opacity,
            marks={i/10: str(i/10) for i in range(1, 11, 2)},
            step=0.1
        ),
    ], id='view-controls', style=style)
