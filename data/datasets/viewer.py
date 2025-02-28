import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import numpy as np
import sys
import torch
sys.path.append("../..")
import utils

# Load dataset instance
from ...configs.common.datasets.change_detection.train.cdd import config
dataset = utils.builders.build_from_config(config['train_dataset'])
transforms_cfg = config['train_dataset']['transforms_cfg']

# Dash app setup
app = dash.Dash(__name__)

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a displayable image."""
    img = tensor.cpu().numpy()
    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3 and img.shape[0] == 3:  # RGB image (C, H, W) -> (H, W, C)
        return np.transpose(img, (1, 2, 0))
    return None  # Non-image tensor

def format_value(value):
    """Format values for display, truncating large tensors."""
    if isinstance(value, torch.Tensor):
        shape = list(value.shape)
        if value.numel() > 10:  # If tensor is too large, only show shape
            return f"Torch Tensor (shape: {shape})"
        return f"Torch Tensor {value.tolist()}"
    return str(value)

# Get available transformations
available_transforms = [
    {'label': t.__class__.__name__, 'value': t.__class__.__name__}
    for t in dataset.transform.transforms  # Use train_dataset.transform instead of transforms_cfg
]

# Layout
app.layout = html.Div([
    dcc.Store(id='current-idx', data=0),  # Store current index in memory

    html.Div([
        html.Button("Previous", id='prev-btn', n_clicks=0, style={'margin-right': '10px'}),
        html.Button("Next", id='next-btn', n_clicks=0)
    ], style={'margin-bottom': '10px'}),

    html.Hr(),
    html.Div(id='datapoint-display'),

    html.Label("Select Active Transformations:"),
    dcc.Dropdown(
        id='transform-selector',
        options=available_transforms,
        multi=True,
        value=[t['value'] for t in available_transforms]
    )
])

@app.callback(
    Output('current-idx', 'data'),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    State('current-idx', 'data'),
    prevent_initial_call=True
)
def update_index(prev_clicks, next_clicks, current_idx):
    """Update the index when buttons are clicked."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_idx

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == 'prev-btn' and current_idx > 0:
        return current_idx - 1
    elif trigger_id == 'next-btn' and current_idx < len(dataset) - 1:
        return current_idx + 1
    return current_idx

@app.callback(
    Output('datapoint-display', 'children'),
    Input('current-idx', 'data'),
    Input('transform-selector', 'value')
)
def update_datapoint(current_idx, selected_transforms):
    """Apply selected transformations and display datapoint details and images."""
    # Load datapoint
    inputs, labels, meta_info = dataset._load_datapoint(current_idx)
    datapoint = {
        'inputs': inputs,
        'labels': labels,
        'meta_info': meta_info,
    }

    # Filter and apply only selected transformations
    filtered_transforms_cfg = {
        'class': Compose,
        'args': {
            'transforms': [
                transform for transform in transforms_cfg['args']['transforms']
                if transform[0].__class__.__name__ in selected_transforms
            ]
        }
    }
    active_transforms = utils.builders.build_from_config(filtered_transforms_cfg)
    datapoint = active_transforms(datapoint)

    display_items = []

    # Display inputs (images)
    for key, value in datapoint['inputs'].items():
        img = tensor_to_image(value)
        if img is not None:
            fig = px.imshow(img)
            fig.update_layout(coloraxis_showscale=False)
            display_items.append(html.Div([html.H5(key), dcc.Graph(figure=fig)]))
        else:
            display_items.append(html.P(f"{key}: {format_value(value)}"))

    # Display labels
    for key, value in datapoint['labels'].items():
        display_items.append(html.P(f"{key}: {format_value(value)}"))

    # Display metadata
    for key, value in datapoint['meta_info'].items():
        display_items.append(html.P(f"{key}: {format_value(value)}"))

    return display_items

if __name__ == '__main__':
    app.run_server(debug=True)
