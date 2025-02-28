import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
import sys
sys.path.append("../..")
import dash_bootstrap_components as dbc
import utils

# Load dataset instance
from ...configs.common.datasets.change_detection.train.cdd import config
train_dataset = utils.builders.build_from_config(config['train_dataset'])

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize index
current_idx = 0

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a displayable image."""
    img = tensor.cpu().numpy()
    if img.ndim == 2:
        return img  # Grayscale image
    elif img.ndim == 3 and img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))  # Convert from CxHxW to HxWxC
    return None

app.layout = html.Div([
    dbc.Row([
        dbc.Col(html.Button("Previous", id='prev-btn', n_clicks=0), width=2),
        dbc.Col(html.Button("Next", id='next-btn', n_clicks=0), width=2)
    ]),
    html.Hr(),
    html.Div(id='datapoint-display'),
    dcc.Dropdown(
        id='transform-selector',
        options=[{'label': t.__class__.__name__, 'value': t.__class__.__name__} for t in train_dataset.transforms_cfg['args']['transforms']],
        multi=True,
        value=[t.__class__.__name__ for t in train_dataset.transforms_cfg['args']['transforms']]
    )
])

@app.callback(
    Output('datapoint-display', 'children'),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    Input('transform-selector', 'value'),
    prevent_initial_call=False
)
def update_datapoint(prev_clicks, next_clicks, selected_transforms):
    global current_idx
    changed_id = dash.callback_context.triggered[0]['prop_id']
    
    if 'prev-btn' in changed_id and current_idx > 0:
        current_idx -= 1
    elif 'next-btn' in changed_id and current_idx < len(train_dataset) - 1:
        current_idx += 1
    
    datapoint = train_dataset[current_idx]
    
    display_items = []
    for key, value in datapoint['inputs'].items():
        img = tensor_to_image(value)
        if img is not None:
            fig = px.imshow(img)
            fig.update_layout(coloraxis_showscale=False)
            display_items.append(html.Div([html.H5(key), dcc.Graph(figure=fig)]))
        else:
            display_items.append(html.P(f"{key}: {value}"))
    
    for key, value in datapoint['labels'].items():
        display_items.append(html.P(f"{key}: {value}"))
    
    for key, value in datapoint['meta_info'].items():
        display_items.append(html.P(f"{key}: {value}"))
    
    return display_items

if __name__ == '__main__':
    app.run_server(debug=True)
