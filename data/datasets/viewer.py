import os
import random
import numpy as np
import torch
import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.express as px
import sys
sys.path.append("../..")
import data
import utils
from data.datasets.pointcloud_utils import (
    create_point_cloud_figure,
    tensor_to_point_cloud
)

# Print debug information to help with path issues
print(f"Current working directory: {os.getcwd()}")
print(f"Repository root (estimated): {os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))}")
print(f"Python sys.path: {sys.path}")

# Find all available dataset configuration files
def get_available_datasets():
    """Get a list of all available dataset configurations."""
    import importlib.util
    import os

    # Adjust the path to be relative to the repository root
    # Since we're running from data/datasets, we need to go up two levels
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dataset_dir = os.path.join(repo_root, "configs/common/datasets/change_detection/train")
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory not found at {dataset_dir}")
        return {}
        
    dataset_configs = {}
    
    for file in os.listdir(dataset_dir):
        if file.endswith('.py') and not file.startswith('_'):
            dataset_name = file[:-3]  # Remove .py extension
            try:
                # Try to import the config to ensure it's valid
                spec = importlib.util.spec_from_file_location(
                    f"configs.common.datasets.change_detection.train.{dataset_name}", 
                    os.path.join(dataset_dir, file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'config'):
                    # Add to the list of valid datasets
                    dataset_configs[dataset_name] = module.config
            except Exception as e:
                print(f"Error loading dataset config {dataset_name}: {e}")
    
    return dataset_configs

# Get all available datasets
AVAILABLE_DATASETS = get_available_datasets()

# Check if we found any datasets
if not AVAILABLE_DATASETS:
    print("Warning: No datasets were found. Adding a fallback dataset for testing.")
    # Add a fallback test dataset for UI testing
    AVAILABLE_DATASETS = {
        "test_dataset": {
            "train_dataset": {
                "class": "TestDataset",
                "args": {
                    "data_root": "./",
                }
            }
        }
    }

# Set a default dataset name (first in the list or fallback to urb3dcd)
DEFAULT_DATASET = next(iter(AVAILABLE_DATASETS.keys())) if AVAILABLE_DATASETS else "urb3dcd"

# Dash app setup
app = dash.Dash(
    __name__, 
    title="Dataset Viewer",
    suppress_callback_exceptions=True  # Add this to handle callbacks to components created by other callbacks
)

# Store for current dataset
current_dataset = None
current_transforms_cfg = None

def tensor_to_image(tensor):
    """Convert a PyTorch tensor to a displayable image."""
    img = tensor.cpu().numpy()
    img = (img-img.min())/(img.max()-img.min())
    if img.ndim == 2:  # Grayscale image
        return img
    elif img.ndim == 3:  # RGB image (C, H, W) -> (H, W, C)
        if img.shape[0] > 3:
            img = img[random.sample(range(img.shape[0]), 3), :, :]
        return np.transpose(img, (1, 2, 0))
    else:
        raise ValueError

def is_point_cloud(data):
    """Check if the input data is a point cloud."""
    if isinstance(data, torch.Tensor):
        # Point cloud typically has shape (N, 3) or more
        return data.ndim == 2 and data.shape[1] >= 3
    return False

def is_3d_dataset(datapoint):
    """Determine if the datapoint is from a 3D dataset."""
    inputs = datapoint['inputs']
    # Check for common point cloud keys
    if 'pc_0' in inputs and 'pc_1' in inputs:
        return True
    return False

def format_value(value):
    """Format values for display, truncating large tensors."""
    if isinstance(value, torch.Tensor):
        shape = list(value.shape)
        if value.numel() > 10:  # If tensor is too large, only show shape
            return f"Torch Tensor (shape: {shape})"
        return f"Torch Tensor {value.tolist()}"
    return str(value)

def get_point_cloud_stats(pc, change_map=None, class_names=None):
    """Get statistical information about a point cloud.

    Args:
        pc: Point cloud tensor of shape (N, 3+)
        change_map: Optional tensor with change classes for each point
        class_names: Optional dictionary mapping class IDs to class names

    Returns:
        Dictionary with point cloud statistics
    """
    if not isinstance(pc, torch.Tensor):
        return {}

    # Convert to numpy for calculations
    pc_np = tensor_to_point_cloud(pc)

    # Basic stats
    stats = {
        "Total Points": len(pc_np),
        "Dimensions": pc_np.shape[1],
        "X Range": f"[{pc_np[:, 0].min():.2f}, {pc_np[:, 0].max():.2f}]",
        "Y Range": f"[{pc_np[:, 1].min():.2f}, {pc_np[:, 1].max():.2f}]",
        "Z Range": f"[{pc_np[:, 2].min():.2f}, {pc_np[:, 2].max():.2f}]",
        "Center": f"[{pc_np[:, 0].mean():.2f}, {pc_np[:, 1].mean():.2f}, {pc_np[:, 2].mean():.2f}]",
    }

    # Add class distribution if change_map is provided
    if change_map is not None:
        unique_classes, class_counts = torch.unique(change_map, return_counts=True)

        # Convert to numpy for display
        unique_classes = unique_classes.cpu().numpy()
        class_counts = class_counts.cpu().numpy()

        # Calculate distribution
        total_points = change_map.numel()
        class_distribution = []

        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / total_points) * 100
            cls_key = cls.item() if hasattr(cls, 'item') else cls

            if class_names and cls_key in class_names:
                class_label = class_names[cls_key]
                class_distribution.append({
                    "class_id": cls_key,
                    "class_name": class_label,
                    "count": int(count),
                    "percentage": percentage
                })
            else:
                class_distribution.append({
                    "class_id": cls_key,
                    "class_name": f"Class {cls_key}",
                    "count": int(count),
                    "percentage": percentage
                })

        stats["class_distribution"] = class_distribution

    return stats

def create_transform_checkboxes(transforms_cfg):
    """Create checkbox components for transforms."""
    if not transforms_cfg or 'args' not in transforms_cfg or 'transforms' not in transforms_cfg['args']:
        return []
        
    return [
        html.Div([
            dcc.Checklist(
                id={'type': 'transform-checkbox', 'index': i},
                options=[
                    {'label': f"{i} - {t[0].__class__.__name__} ({[key_pair for key_pair in t[1]]})", 'value': i}
                ],
                value=[i]
            )
        ])
        for i, t in enumerate(transforms_cfg['args']['transforms'])
    ]

def create_app_layout():
    """Create the application layout."""
    # Generate dataset dropdown options
    dataset_options = [{'label': name, 'value': name} for name in AVAILABLE_DATASETS.keys()]
    
    return html.Div([
        # Header row with title and dataset dropdown
        html.Div([
            html.Div([
                html.H1("Dataset Viewer", style={'margin-bottom': '0px', 'margin-right': '20px', 'display': 'inline-block'}),
            ], style={'display': 'inline-block', 'vertical-align': 'middle'}),
            
            html.Div([
                html.Label("Select Dataset:", style={'font-weight': 'bold', 'margin-right': '10px', 'display': 'inline-block'}),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=dataset_options,
                    value=DEFAULT_DATASET,
                    style={'width': '300px', 'display': 'inline-block'}
                ),
            ], style={'display': 'inline-block', 'vertical-align': 'middle'}),
        ], style={'display': 'flex', 'align-items': 'center', 'justify-content': 'flex-start', 'padding': '10px', 'background-color': '#f0f0f0', 'margin-bottom': '10px'}),

        dcc.Store(id='current-idx', data=0),  # Store current index in memory
        dcc.Store(id='camera-view', data=None),  # Store camera view for syncing
        dcc.Store(id='current-dataset-info', data={'name': DEFAULT_DATASET}),  # Store current dataset info
        dcc.Store(id='is-3d-dataset', data=False),  # Store whether current dataset is 3D

        # Main content area
        html.Div([
            # Control panel (left sidebar)
            html.Div(id='control', children=[
                html.Div(id='navigation', children=[
                    html.P(id='index', children='index=0', style={'margin-bottom': '10px', 'font-weight': 'bold'}),
                    html.Div([
                        html.Button("⏮ Prev", id='prev-btn', n_clicks=0,
                                  style={'margin-right': '10px', 'background-color': '#e7e7e7', 'border': 'none', 'padding': '10px 20px'}),
                        html.Button("Next ⏭", id='next-btn', n_clicks=0,
                                  style={'background-color': '#e7e7e7', 'border': 'none', 'padding': '10px 20px'}),
                    ], style={'margin-bottom': '20px'}),
                    html.P(id='total-samples', children="Total samples: 0", style={'margin-bottom': '20px'}),
                ]),
                html.Div(id='transforms', children=[
                    html.Label("Select Active Transformations:", style={'font-weight': 'bold'}),
                    html.Div(id='transform-checkboxes', style={'margin-bottom': '20px', 'max-height': '200px', 'overflow-y': 'auto'})
                ]),
                # 3D View Options - include in initial layout but hidden by default
                html.Div(id='view-controls', children=[
                    html.Label("3D View Options:", style={'font-weight': 'bold'}),
                    html.Label("Point Size:"),
                    dcc.Slider(
                        id='point-size-slider',
                        min=1,
                        max=5,
                        step=0.5,
                        value=2,
                        marks={i: str(i) for i in range(1, 6)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Label("Point Opacity:"),
                    dcc.Slider(
                        id='point-opacity-slider',
                        min=0.1,
                        max=1.0,
                        step=0.1,
                        value=0.8,
                        marks={i/10: str(i/10) for i in range(1, 11)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'display': 'none'}),  # Hidden by default
                html.Div(id='dataset-info', children=[]),
            ], style={'width': '18%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '10px', 'background-color': '#f9f9f9'}),

            # Main display area (right side)
            html.Div(id='datapoint-display', style={'width': '80%', 'display': 'inline-block', 'padding-left': '10px'})
        ], style={'display': 'flex', 'width': '100%'})
    ], style={'font-family': 'Arial, sans-serif', 'margin': '0', 'padding': '0'})

# Set the app layout
app.layout = create_app_layout()

# Callback to load dataset and update transforms when dataset selection changes
@app.callback(
    Output('current-dataset-info', 'data'),
    Output('transform-checkboxes', 'children'),
    Output('total-samples', 'children'),
    Output('dataset-info', 'children'),
    Output('current-idx', 'data', allow_duplicate=True),  # Allow duplicate to resolve the conflict
    Output('is-3d-dataset', 'data', allow_duplicate=True),  # Also reset 3D flag when dataset changes
    Input('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def load_dataset(dataset_name):
    """Load the selected dataset and update the UI accordingly."""
    global current_dataset, current_transforms_cfg
    
    if dataset_name not in AVAILABLE_DATASETS:
        return (
            {'name': None, 'class_labels': {}}, 
            [], 
            "Total samples: 0", 
            html.H4("No dataset loaded"), 
            0,
            False  # Not a 3D dataset
        )
    
    # Load dataset configuration
    dataset_cfg = AVAILABLE_DATASETS[dataset_name]['train_dataset']
    
    # Handle test dataset for UI testing
    if dataset_cfg.get('class') == 'TestDataset':
        # Return a placeholder for testing
        return (
            {'name': dataset_name, 'class_labels': {0: 'No Change', 1: 'Change'}},
            [],
            "Total samples: 0 (Test Dataset)",
            [
                html.H4("Test Dataset", style={'margin-top': '30px'}),
                html.P("This is a placeholder for testing the UI when no real datasets are found.")
            ],
            0,
            False  # Not a 3D dataset
        )
    
    # Adjust data_root path to be relative to the repository root
    if 'data_root' in dataset_cfg['args']:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # Make the path absolute by joining with repo_root if it's a relative path
        data_root = dataset_cfg['args']['data_root']
        if not os.path.isabs(data_root):
            data_root = os.path.join(repo_root, data_root)
        dataset_cfg['args']['data_root'] = data_root
    
    try:
        # Get transforms configuration
        transforms_cfg = dataset_cfg['args'].get(
            'transforms_cfg', 
            {'class': data.transforms.Compose, 'args': {'transforms': []}}
        )
        
        # Build dataset
        dataset = utils.builders.build_from_config(dataset_cfg)
        current_dataset = dataset
        current_transforms_cfg = transforms_cfg
        
        # Create transform checkboxes
        transform_checkboxes = create_transform_checkboxes(transforms_cfg)
        
        # Check if the dataset has specific attributes for visualization
        has_class_labels = hasattr(dataset, 'INV_OBJECT_LABEL') or hasattr(dataset, 'CLASS_LABELS')
        class_labels = {}
        
        if hasattr(dataset, 'INV_OBJECT_LABEL'):
            class_labels = dataset.INV_OBJECT_LABEL
        elif hasattr(dataset, 'CLASS_LABELS'):
            class_labels = {v: k for k, v in dataset.CLASS_LABELS.items()}
        
        # Create dataset info display
        dataset_info = [
            html.H4("Dataset Information:", style={'margin-top': '30px'}),
            html.P(f"Dataset Type: {dataset.__class__.__name__}"),
        ]
        
        if has_class_labels:
            dataset_info.extend([
                html.P("Change Classes:", style={'font-weight': 'bold'}),
                html.Div([
                    html.P(f"{class_id}: {class_name}", style={'margin-left': '20px'})
                    for class_id, class_name in class_labels.items()
                ])
            ])
        
        # Try to determine if this is a 3D dataset based on the class name
        is_3d = 'Point' in dataset.__class__.__name__ or '3D' in dataset.__class__.__name__
        
        return (
            {'name': dataset_name, 'class_labels': class_labels},
            transform_checkboxes,
            f"Total samples: {len(dataset)}",
            dataset_info,
            0,  # Reset index to 0
            is_3d  # Set initial 3D flag based on dataset class name
        )
    except Exception as e:
        # Log the error and return an error message
        print(f"Error loading dataset {dataset_name}: {e}")
        return (
            {'name': None, 'class_labels': {}},
            [],
            "Total samples: 0",
            [
                html.H4("Error Loading Dataset", style={'color': 'red', 'margin-top': '30px'}),
                html.P(f"Error: {str(e)}")
            ],
            0,
            False  # Not a 3D dataset
        )

# Add a separate callback to update the index display when the current index changes
@app.callback(
    Output('index', 'children'),
    Input('current-idx', 'data')
)
def update_index_display(current_idx):
    """Update the index display when the index changes."""
    return f"Index: {current_idx}"

@app.callback(
    Output('current-idx', 'data', allow_duplicate=True),
    Input('prev-btn', 'n_clicks'),
    Input('next-btn', 'n_clicks'),
    State('current-idx', 'data'),
    prevent_initial_call=True
)
def update_index(prev_clicks, next_clicks, current_idx):
    """Update the index when buttons are clicked."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if current_dataset is None:
        return 0

    if trigger_id == 'prev-btn' and current_idx > 0:
        current_idx = current_idx - 1
    elif trigger_id == 'next-btn' and current_idx < len(current_dataset) - 1:
        current_idx = current_idx + 1
    else:
        return dash.no_update
        
    return current_idx

@app.callback(
    Output('camera-view', 'data'),
    Input({'type': 'point-cloud-graph', 'index': ALL}, 'relayoutData'),
    State('camera-view', 'data'),
    prevent_initial_call=True
)
def sync_camera_views(relayout_data_list, current_camera):
    """Synchronize camera views across all point cloud graphs."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_camera

    # Get the relayout data from the trigger
    for i, data in enumerate(relayout_data_list):
        if data and 'scene.camera' in data:
            return data['scene.camera']

    return current_camera

@app.callback(
    Output({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
    Input('camera-view', 'data'),
    State({'type': 'point-cloud-graph', 'index': ALL}, 'figure'),
    prevent_initial_call=True
)
def update_camera_views(camera_data, figures):
    """Update all point cloud figures with the synchronized camera view."""
    if not camera_data or not figures:
        return [dash.no_update] * len(figures)

    updated_figures = []
    for fig in figures:
        if fig:
            # Make a copy of the figure to avoid modifying the original
            updated_fig = fig.copy()
            # Update camera
            updated_fig['layout']['scene']['camera'] = camera_data
            updated_figures.append(updated_fig)
        else:
            updated_figures.append(dash.no_update)

    return updated_figures

@app.callback(
    Output('datapoint-display', 'children'),
    Output('is-3d-dataset', 'data', allow_duplicate=True),
    Input('current-idx', 'data'),
    Input({'type': 'transform-checkbox', 'index': dash.ALL}, 'value'),
    Input('point-size-slider', 'value'),
    Input('point-opacity-slider', 'value'),
    Input('current-dataset-info', 'data'),
    State('is-3d-dataset', 'data'),
    prevent_initial_call=True
)
def update_datapoint(current_idx, selected_transform_indices, point_size, point_opacity, dataset_info, is_3d):
    """Apply selected transformations and display datapoint details and images/point clouds."""
    if current_dataset is None or dataset_info['name'] is None:
        return html.Div(html.H3("No dataset loaded. Please select a dataset from the dropdown.")), False
    
    # Handle test dataset case
    if dataset_info['name'] == 'test_dataset':
        return html.Div([
            html.H2("Test Dataset Viewer", style={'text-align': 'center'}),
            html.P("This is a placeholder UI for testing when no real datasets are available."),
            html.P("Please ensure your config directories are correctly set up and accessible."),
            html.Div(style={'background-color': '#f8f9fa', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'}, children=[
                html.H3("Troubleshooting Tips:"),
                html.Ul([
                    html.Li("Check that the repository structure is correct"),
                    html.Li("Verify that 'configs/common/datasets/change_detection/train' exists in your repository"),
                    html.Li("Make sure dataset config files have the expected format"),
                    html.Li("Run the script from the repository root instead of the data/datasets directory")
                ])
            ])
        ]), False
    
    selected_transform_indices = [i[0] for i in selected_transform_indices if i]

    try:
        # Load datapoint
        inputs, labels, meta_info = current_dataset._load_datapoint(current_idx)
        datapoint = {
            'inputs': inputs,
            'labels': labels,
            'meta_info': meta_info,
        }

        # Filter transforms by selected indices
        if current_transforms_cfg and 'args' in current_transforms_cfg and 'transforms' in current_transforms_cfg['args']:
            filtered_transforms_cfg = {
                'class': data.transforms.Compose,
                'args': {
                    'transforms': [
                        transform for i, transform in enumerate(current_transforms_cfg['args']['transforms'])
                        if i in selected_transform_indices
                    ],
                },
            }

            # Build the transformation pipeline with only selected transforms
            active_transforms = utils.builders.build_from_config(filtered_transforms_cfg)
            datapoint = active_transforms(datapoint)

        # Get class labels
        class_labels = dataset_info['class_labels']
        
        # Check if we're dealing with a 3D dataset
        is_3d_datapoint = is_3d_dataset(datapoint)
        
        if is_3d_datapoint:
            return display_3d_datapoint(datapoint, point_size, point_opacity, class_labels), True
        else:
            return display_2d_datapoint(datapoint), False
    except Exception as e:
        # Handle errors that might occur during datapoint loading or processing
        return html.Div([
            html.H3("Error Loading Datapoint", style={'color': 'red'}),
            html.P(f"An error occurred: {str(e)}"),
            html.P("This could be due to missing data, incorrect paths, or incompatible data formats.")
        ]), False

def display_2d_datapoint(datapoint):
    """Display a 2D image datapoint."""
    img_1 = tensor_to_image(datapoint['inputs']['img_1'])
    img_2 = tensor_to_image(datapoint['inputs']['img_2'])
    change_map = tensor_to_image(datapoint['labels']['change_map'])

    input_fig_1 = px.imshow(img_1)
    input_fig_1.update_layout(coloraxis_showscale=False, title='Image 1')
    input_fig_2 = px.imshow(img_2)
    input_fig_2.update_layout(coloraxis_showscale=False, title='Image 2')
    change_map_fig = px.imshow(change_map, color_continuous_scale='viridis')
    change_map_fig.update_layout(coloraxis_showscale=False, title='Change Map')

    return html.Div([
        html.H2("2D Change Detection Visualization", style={'text-align': 'center'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=input_fig_1),
                html.P(f"Shape: {img_1.shape}"),
                html.P(f"Type: {img_1.dtype}"),
                html.P(f"Range: [{img_1.min():.4f}, {img_1.max():.4f}]"),
                dcc.Graph(figure=input_fig_2),
                html.P(f"Shape: {img_2.shape}"),
                html.P(f"Type: {img_2.dtype}"),
                html.P(f"Range: [{img_2.min():.4f}, {img_2.max():.4f}]"),
            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                dcc.Graph(figure=change_map_fig),
                html.P(f"Shape: {change_map.shape}"),
                html.P(f"Type: {change_map.dtype}"),
                html.P(f"Range: [{change_map.min():.4f}, {change_map.max():.4f}]"),
                html.H4("Metadata"),
                html.Div([
                    html.P(f"{key}: {format_value(value)}")
                    for key, value in datapoint['meta_info'].items()
                ], style={'max-height': '300px', 'overflow-y': 'auto'})
            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'}),
        ], style={'display': 'flex'}),
    ])

def display_3d_datapoint(datapoint, point_size=2, point_opacity=0.8, class_labels=None):
    """Display a 3D point cloud datapoint."""
    pc_0 = datapoint['inputs']['pc_0']
    pc_1 = datapoint['inputs']['pc_1']
    change_map = datapoint['labels']['change_map']

    # Get statistics for both point clouds
    pc_0_stats = get_point_cloud_stats(pc_0)
    pc_1_stats = get_point_cloud_stats(pc_1, change_map=None)

    # Get change map stats with class distribution
    change_stats = get_point_cloud_stats(pc_1, change_map=change_map, class_names=class_labels)

    # Create point cloud figures
    pc_0_fig = create_point_cloud_figure(
        pc_0,
        title='Point Cloud 1',
        point_size=point_size,
        opacity=point_opacity
    )
    pc_1_fig = create_point_cloud_figure(
        pc_1,
        title='Point Cloud 2',
        point_size=point_size,
        opacity=point_opacity
    )

    # Create change map figure (colorized by change class)
    change_map_fig = create_point_cloud_figure(
        pc_1,
        colors=change_map,
        title='Change Map',
        colorscale='Viridis',
        point_size=point_size,
        opacity=point_opacity,
        colorbar_title="Change Class"
    )

    # Create class distribution components
    class_distribution = []
    if 'class_distribution' in change_stats:
        for class_info in change_stats['class_distribution']:
            class_distribution.append(
                html.P(f"{class_info['class_name']}: {class_info['count']} points ({class_info['percentage']:.1f}%)")
            )

    # Create legend for change map classes
    legend_items = []
    if class_labels:
        for class_id, class_name in class_labels.items():
            legend_items.append(html.Div([
                html.Div(style={
                    'background-color': f'#{class_id * 30:02x}{255 - class_id * 30:02x}{255:02x}',
                    'width': '20px',
                    'height': '20px',
                    'display': 'inline-block',
                    'margin-right': '5px',
                    'vertical-align': 'middle'
                }),
                html.Span(f"Class {class_id}: {class_name}", style={'vertical-align': 'middle'})
            ], style={'margin-bottom': '5px'}))

    return html.Div([
        html.H2("3D Point Cloud Change Detection Visualization", style={'text-align': 'center'}),
        html.P("Point clouds are synchronized - drag one to adjust all views simultaneously",
               style={'text-align': 'center', 'font-style': 'italic'}),

        # Main content with visualizations and class distribution
        html.Div([
            # Point cloud visualizations (equal width)
            html.Div([
                html.Div([
                    dcc.Graph(
                        id={'type': 'point-cloud-graph', 'index': 0},
                        figure=pc_0_fig,
                        config={'scrollZoom': True},
                        style={'height': '500px'}
                    ),
                    html.H4("Point Cloud 1 Statistics"),
                    html.Div([
                        html.P(f"{key}: {value}")
                        for key, value in pc_0_stats.items() if key != 'class_distribution'
                    ]),
                ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'box-sizing': 'border-box'}),

                html.Div([
                    dcc.Graph(
                        id={'type': 'point-cloud-graph', 'index': 1},
                        figure=pc_1_fig,
                        config={'scrollZoom': True},
                        style={'height': '500px'}
                    ),
                    html.H4("Point Cloud 2 Statistics"),
                    html.Div([
                        html.P(f"{key}: {value}")
                        for key, value in pc_1_stats.items() if key != 'class_distribution'
                    ]),
                ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'box-sizing': 'border-box'}),

                html.Div([
                    dcc.Graph(
                        id={'type': 'point-cloud-graph', 'index': 2},
                        figure=change_map_fig,
                        config={'scrollZoom': True},
                        style={'height': '500px'}
                    ),
                    html.H4("Change Class Distribution"),
                    html.Div(class_distribution, style={'max-height': '200px', 'overflow-y': 'auto'}),
                ], style={'width': '33%', 'display': 'inline-block', 'vertical-align': 'top', 'box-sizing': 'border-box'}),
            ], style={'display': 'flex', 'width': '100%'}),

            # Legend section below
            html.Div([
                html.H4("Change Classes Legend", style={'margin-top': '20px', 'margin-bottom': '10px'}),
                html.Div(legend_items, style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}),
            ]),
        ]),

        # Metadata section
        html.Div([
            html.H4("Metadata"),
            html.Div([
                html.P(f"{key}: {format_value(value)}")
                for key, value in datapoint['meta_info'].items()
            ], style={'max-height': '200px', 'overflow-y': 'auto'})
        ], style={'margin-top': '20px'}),
    ])

@app.callback(
    Output('view-controls', 'style'),
    Input('is-3d-dataset', 'data')
)
def update_view_controls(is_3d):
    """Show or hide 3D view controls based on the type of dataset currently displayed."""
    # Set visibility style based on is_3d flag
    return {'display': 'block', 'margin-top': '20px'} if is_3d else {'display': 'none'}

if __name__ == '__main__':
    app.run_server(debug=True)
