"""App layout components for the dataset viewer."""
from dash import dcc, html
from data.viewer.layout.controls.dataset import create_dataset_selector, create_reload_button
from data.viewer.layout.controls.navigation import create_navigation_controls
from data.viewer.layout.controls.controls_3d import create_3d_controls


def create_app_layout(available_datasets):
    """Create the main application layout.
    
    Args:
        available_datasets: Dictionary of available datasets
        
    Returns:
        Dash layout component
    """
    # Initialize Dash components
    layout = html.Div([
        # Hidden stores for keeping track of state
        dcc.Store(id='dataset-info', data={}),
        dcc.Store(id='is-3d-dataset', data=False),

        # Header
        html.Div([
            html.H1("Dataset Viewer", style={'text-align': 'center', 'margin-bottom': '20px'}),

            # Dataset selector and reload button
            html.Div([
                create_dataset_selector(available_datasets),
                create_reload_button()
            ], style={'display': 'flex', 'align-items': 'flex-end'}),

            # Navigation controls
            create_navigation_controls()
        ], style={'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px', 'margin-bottom': '20px'}),

        # Main content area
        html.Div([
            # Left sidebar with controls and info
            html.Div([
                # Dataset info section
                html.Div(id='dataset-info-display', style={'margin-bottom': '20px'}),
                
                # Transforms section
                html.Div(id='transforms-section', style={'margin-bottom': '20px'}),

                # 3D View Controls - initially hidden, shown only for 3D datasets
                create_3d_controls(visible=False)
            ], style={'width': '25%', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '5px'}),

            # Right main display area
            html.Div([
                html.Div(id='datapoint-display', style={'padding': '10px'})
            ], style={'width': '75%', 'padding': '20px', 'background-color': '#ffffff', 'border-radius': '5px'})
        ], style={'display': 'flex', 'gap': '20px'})
    ])

    return layout
