import os
import torch
import numpy as np
import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import sys
sys.path.append("../..")
import data
import utils

# Load dataset instance
from configs.common.datasets.change_detection.train.cdd import config
dataset_cfg = config['train_dataset']
dataset_cfg['args']['data_root'] = os.path.relpath(dataset_cfg['args']['data_root'], start="./data/datasets")
transforms_cfg = dataset_cfg['args']['transforms_cfg']
dataset = utils.builders.build_from_config(dataset_cfg)

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

# Generate dropdown options with transform indices
available_transforms = [
    {
        'label': f"{i} - {t[0].__class__.__name__} ({[key_pair for key_pair in t[1]]})",
        'value': i
    }
    for i, t in enumerate(transforms_cfg['args']['transforms'])
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

    html.Label("Select Active Transformations (Ordered by Index):"),
    dcc.Dropdown(
        id='transform-selector',
        options=available_transforms,
        multi=True,
        value=[t['value'] for t in available_transforms]  # Default to all selected
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
def update_datapoint(current_idx, selected_transform_indices):
    """Apply selected transformations and display datapoint details and images."""
    # Load datapoint
    inputs, labels, meta_info = dataset._load_datapoint(current_idx)
    datapoint = {
        'inputs': inputs,
        'labels': labels,
        'meta_info': meta_info,
    }

    # Filter transforms by selected indices
    filtered_transforms_cfg = {
        'class': data.transforms.Compose,
        'args': {
            'transforms': [
                transform for i, transform in enumerate(transforms_cfg['args']['transforms'])
                if i in selected_transform_indices
            ]
        }
    }
    
    # Build the transformation pipeline with only selected transforms
    active_transforms = utils.builders.build_from_config(filtered_transforms_cfg)
    datapoint = active_transforms(datapoint)

    # Layout for change detection dataset
    img_1 = tensor_to_image(datapoint['inputs']['img_1'])
    img_2 = tensor_to_image(datapoint['inputs']['img_2'])
    change_map = tensor_to_image(datapoint['labels']['change_map'])

    input_fig_1 = px.imshow(img_1)
    input_fig_1.update_layout(coloraxis_showscale=False, title='Image 1')
    input_fig_2 = px.imshow(img_2)
    input_fig_2.update_layout(coloraxis_showscale=False, title='Image 2')
    change_map_fig = px.imshow(change_map, color_continuous_scale='viridis')
    change_map_fig.update_layout(coloraxis_showscale=False, title='Change Map')

    # Display section
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=input_fig_1)]),
            html.Div([dcc.Graph(figure=input_fig_2)])
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(figure=change_map_fig)
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Hr(),
        html.Div([
            html.H5("Metadata"),
            *[html.P(f"{key}: {format_value(value)}") for key, value in datapoint['meta_info'].items()]
        ])
    ])

if __name__ == '__main__':
    app.run_server(debug=True)
