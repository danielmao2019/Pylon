"""UI components for the dataset viewer."""
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import torch
import traceback
from data.viewer.utils.dataset_utils import format_value, tensor_to_image, get_point_cloud_stats

def display_2d_datapoint(datapoint):
    """Display a 2D image datapoint."""
    # Check if the inputs have the expected structure
    img_1 = datapoint['inputs'].get('img_1')
    img_2 = datapoint['inputs'].get('img_2')
    change_map = datapoint['labels'].get('change_map')
    
    # Verify that all required data is present and has the correct type
    error_messages = []
    
    if img_1 is None:
        error_messages.append("Image 1 (img_1) is missing")
    elif not isinstance(img_1, torch.Tensor):
        error_messages.append(f"Image 1 (img_1) has unexpected type: {type(img_1).__name__}")
        
    if img_2 is None:
        error_messages.append("Image 2 (img_2) is missing")
    elif not isinstance(img_2, torch.Tensor):
        error_messages.append(f"Image 2 (img_2) has unexpected type: {type(img_2).__name__}")
        
    if change_map is None:
        error_messages.append("Change map is missing")
    elif not isinstance(change_map, torch.Tensor):
        error_messages.append(f"Change map has unexpected type: {type(change_map).__name__}")
    
    # If any errors were found, display them
    if error_messages:
        return html.Div([
            html.H3("Error Displaying 2D Image Data", style={'color': 'red'}),
            html.P("The dataset structure doesn't match the expected format:"),
            html.Ul([html.Li(msg) for msg in error_messages]),
            html.P("Datapoint structure:"),
            html.Div([
                html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
                html.Pre(f"Meta info: {format_value(datapoint.get('meta_info', {}))}")
            ], style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'})
        ])

    # Convert tensors to displayable images
    try:
        img_1_display = tensor_to_image(img_1)
        img_2_display = tensor_to_image(img_2)
        change_map_display = tensor_to_image(change_map)
    except Exception as e:
        return html.Div([
            html.H3("Error Converting Tensors to Images", style={'color': 'red'}),
            html.P(f"Error: {str(e)}"),
            html.P("This may be due to incompatible data formats or invalid tensor shapes.")
        ])

    # Create plotly figures
    try:
        input_fig_1 = px.imshow(img_1_display)
        input_fig_1.update_layout(coloraxis_showscale=False, title='Image 1')
        input_fig_2 = px.imshow(img_2_display)
        input_fig_2.update_layout(coloraxis_showscale=False, title='Image 2')
        change_map_fig = px.imshow(change_map_display, color_continuous_scale='viridis')
        change_map_fig.update_layout(coloraxis_showscale=False, title='Change Map')
    except Exception as e:
        return html.Div([
            html.H3("Error Creating Image Figures", style={'color': 'red'}),
            html.P(f"Error: {str(e)}"),
            html.P("This may be due to incompatible data formats.")
        ])

    return html.Div([
        html.H2("2D Change Detection Visualization", style={'text-align': 'center'}),
        html.Div([
            html.Div([
                dcc.Graph(figure=input_fig_1),
                html.P(f"Shape: {img_1_display.shape}"),
                html.P(f"Type: {img_1_display.dtype}"),
                html.P(f"Range: [{img_1_display.min():.4f}, {img_1_display.max():.4f}]"),
                dcc.Graph(figure=input_fig_2),
                html.P(f"Shape: {img_2_display.shape}"),
                html.P(f"Type: {img_2_display.dtype}"),
                html.P(f"Range: [{img_2_display.min():.4f}, {img_2_display.max():.4f}]"),
            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
            html.Div([
                dcc.Graph(figure=change_map_fig),
                html.P(f"Shape: {change_map_display.shape}"),
                html.P(f"Type: {change_map_display.dtype}"),
                html.P(f"Range: [{change_map_display.min():.4f}, {change_map_display.max():.4f}]"),
                html.H4("Metadata"),
                html.Div([
                    html.P(f"{key}: {format_value(value)}")
                    for key, value in datapoint.get('meta_info', {}).items()
                ], style={'max-height': '300px', 'overflow-y': 'auto'})
            ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'}),
        ], style={'display': 'flex'}),
    ])

def display_3d_datapoint(datapoint, point_size=2, point_opacity=0.8, class_names=None):
    """Display a 3D point cloud datapoint."""
    # Check if the inputs have the expected structure
    pc_0 = datapoint['inputs'].get('pc_0')
    pc_1 = datapoint['inputs'].get('pc_1') 
    change_map = datapoint['labels'].get('change_map')
    
    # Verify that all required data is present and has the correct type
    error_messages = []
    
    # For pc_0 (first point cloud)
    if pc_0 is None:
        # Check alternative names
        pc_0 = datapoint['inputs'].get('point_cloud')
        if pc_0 is None:
            error_messages.append("Point cloud 1 (pc_0) is missing")
    
    if pc_0 is not None and not isinstance(pc_0, torch.Tensor):
        error_messages.append(f"Point cloud 1 (pc_0) has unexpected type: {type(pc_0).__name__}")
    
    # pc_1 is optional for some datasets
    if pc_1 is not None and not isinstance(pc_1, torch.Tensor):
        error_messages.append(f"Point cloud 2 (pc_1) has unexpected type: {type(pc_1).__name__}")
    
    # change_map is optional for some datasets
    if change_map is not None and not isinstance(change_map, torch.Tensor):
        error_messages.append(f"Change map has unexpected type: {type(change_map).__name__}")
    
    # If any errors were found, display them
    if error_messages:
        return html.Div([
            html.H3("Error Displaying 3D Point Cloud Data", style={'color': 'red'}),
            html.P("The dataset structure doesn't match the expected format:"),
            html.Ul([html.Li(msg) for msg in error_messages]),
            html.P("Datapoint structure:"),
            html.Div([
                html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
                html.Pre(f"Meta info: {format_value(datapoint.get('meta_info', {}))}"),
            ], style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'})
        ])
    
    try:
        # Prepare point clouds for display
        if pc_0 is not None:
            # Handle different point cloud formats
            if len(pc_0.shape) == 3 and pc_0.shape[1] == 1:
                # Shape is [N, 1, 3] - remove the middle dimension
                pc_0 = pc_0.squeeze(1)
            
            # Ensure point cloud is 2D tensor with shape [N, 3]
            if len(pc_0.shape) != 2 or pc_0.shape[1] != 3:
                error_messages.append(f"Point cloud 1 has unexpected shape: {pc_0.shape}. Expected [N, 3]")
                
        if pc_1 is not None:
            if len(pc_1.shape) == 3 and pc_1.shape[1] == 1:
                pc_1 = pc_1.squeeze(1)
            
            if len(pc_1.shape) != 2 or pc_1.shape[1] != 3:
                error_messages.append(f"Point cloud 2 has unexpected shape: {pc_1.shape}. Expected [N, 3]")
    
        # If errors were found in the shapes, display them
        if error_messages:
            return html.Div([
                html.H3("Error Processing Point Cloud Data", style={'color': 'red'}),
                html.P("The point cloud data has unexpected dimensions:"),
                html.Ul([html.Li(msg) for msg in error_messages]),
                html.P("Datapoint structure:"),
                html.Div([
                    html.P(f"pc_0 shape: {pc_0.shape if pc_0 is not None else 'None'}"),
                    html.P(f"pc_1 shape: {pc_1.shape if pc_1 is not None else 'None'}"),
                ], style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'})
            ])
        
        # Prepare figure with both point clouds
        if pc_0 is not None and pc_1 is not None:
            # Case: We have two point clouds (change detection)
            point_clouds = []
            
            # First point cloud (pc_0)
            pc_0_numpy = pc_0.detach().cpu().numpy()
            
            if change_map is not None:
                # Use change map for coloring if available
                change_values = change_map.detach().cpu().numpy().flatten()
                point_clouds.append(
                    go.Scatter3d(
                        x=pc_0_numpy[:, 0], 
                        y=pc_0_numpy[:, 1], 
                        z=pc_0_numpy[:, 2],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            opacity=point_opacity,
                            color=change_values,
                            colorscale='Viridis',
                            colorbar=dict(
                                title='Change',
                                thickness=20
                            )
                        ),
                        name='Point Cloud 1 with Change'
                    )
                )
            else:
                # No change map, use default coloring
                point_clouds.append(
                    go.Scatter3d(
                        x=pc_0_numpy[:, 0], 
                        y=pc_0_numpy[:, 1], 
                        z=pc_0_numpy[:, 2],
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            opacity=point_opacity,
                            color='blue'
                        ),
                        name='Point Cloud 1'
                    )
                )
            
            # Second point cloud (pc_1)
            pc_1_numpy = pc_1.detach().cpu().numpy()
            point_clouds.append(
                go.Scatter3d(
                    x=pc_1_numpy[:, 0], 
                    y=pc_1_numpy[:, 1], 
                    z=pc_1_numpy[:, 2],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        opacity=point_opacity,
                        color='red'
                    ),
                    name='Point Cloud 2'
                )
            )
            
            # Create layout with both point clouds
            fig = go.Figure(data=point_clouds)
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                legend=dict(x=0, y=1)
            )
            
            # Get stats for both point clouds
            pc_0_stats = get_point_cloud_stats(pc_0, change_map, class_names)
            pc_1_stats = get_point_cloud_stats(pc_1, None, class_names)
            
            stats_content = html.Div([
                html.Div([
                    html.H4("Point Cloud 1 Statistics"),
                    html.Ul([html.Li(f"{k}: {v}") for k, v in pc_0_stats.items()])
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H4("Point Cloud 2 Statistics"),
                    html.Ul([html.Li(f"{k}: {v}") for k, v in pc_1_stats.items()])
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '20px'}),
            ])
            
        elif pc_0 is not None:
            # Case: We only have one point cloud
            pc_0_numpy = pc_0.detach().cpu().numpy()
            
            if change_map is not None:
                # Use change map or class labels for coloring if available
                color_values = change_map.detach().cpu().numpy().flatten()
                colorscale = 'Viridis'
                color_title = 'Change/Class'
            else:
                # Default coloring by height (Z coordinate)
                color_values = pc_0_numpy[:, 2]
                colorscale = 'Viridis'
                color_title = 'Height'
            
            # Create figure with single point cloud
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=pc_0_numpy[:, 0], 
                    y=pc_0_numpy[:, 1], 
                    z=pc_0_numpy[:, 2],
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        opacity=point_opacity,
                        color=color_values,
                        colorscale=colorscale,
                        colorbar=dict(
                            title=color_title,
                            thickness=20
                        )
                    ),
                    name='Point Cloud'
                )
            ])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z',
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Get stats for the point cloud
            pc_0_stats = get_point_cloud_stats(pc_0, change_map, class_names)
            
            stats_content = html.Div([
                html.H4("Point Cloud Statistics"),
                html.Ul([html.Li(f"{k}: {v}") for k, v in pc_0_stats.items()])
            ])
        else:
            # Case: No valid point clouds found
            return html.Div([
                html.H3("Error: No Valid Point Cloud Data", style={'color': 'red'}),
                html.P("The dataset doesn't contain valid point cloud data."),
                html.P("Datapoint structure:"),
                html.Div([
                    html.P(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                    html.P(f"Labels keys: {list(datapoint['labels'].keys())}"),
                ], style={'background-color': '#f0f0f0', 'padding': '10px', 'border-radius': '5px'})
            ])
            
        # Return the complete 3D view
        return html.Div([
            html.H2("3D Point Cloud Visualization", style={'text-align': 'center'}),
            dcc.Graph(
                figure=fig,
                style={'height': '70vh'}
            ),
            stats_content,
            html.H4("Metadata"),
            html.Div([
                html.P(f"{key}: {format_value(value)}")
                for key, value in datapoint.get('meta_info', {}).items()
            ], style={'max-height': '300px', 'overflow-y': 'auto'})
        ])
            
    except Exception as e:
        error_traceback = traceback.format_exc()
        return html.Div([
            html.H3("Error Visualizing Point Cloud", style={'color': 'red'}),
            html.P(f"Error: {str(e)}"),
            html.Pre(error_traceback, style={
                'background-color': '#ffeeee',
                'padding': '10px',
                'border-radius': '5px',
                'max-height': '300px',
                'overflow-y': 'auto'
            })
        ])

def create_transform_checkboxes(transforms):
    """Create checkbox components for transforms."""
    if not transforms:
        return []
        
    return [
        html.Div([
            dcc.Checklist(
                id={'type': 'transform-checkbox', 'index': i},
                options=[
                    {'label': f"{transform.get('class', 'Unknown').__name__ if hasattr(transform.get('class', 'Unknown'), '__name__') else 'Unknown'}", 'value': i}
                ],
                value=[i]
            )
        ], style={'margin-bottom': '5px'})
        for i, transform in enumerate(transforms)
    ] 