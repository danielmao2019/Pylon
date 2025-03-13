"""Display-related callbacks for the viewer."""
from dash import Input, Output, State
import html
import traceback
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint
from data.viewer.callbacks.registry import callback


@callback(
    outputs=[
        Output('datapoint-display', 'children'),
        Output('is-3d-dataset', 'data')
    ],
    inputs=[
        Input('dataset-info', 'data'),
        Input('datapoint-index-slider', 'value'),
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value')
    ],
    states=[State('is-3d-dataset', 'data')],
    group="display"
)
def update_datapoint(dataset_info, datapoint_idx, point_size, point_opacity, is_3d_prev):
    """
    Update the displayed datapoint based on the slider value.
    Also handles 3D point cloud visualization settings.
    """
    if dataset_info is None or dataset_info == {}:
        return html.Div("No dataset loaded."), False

    try:
        dataset_name = dataset_info.get('name', 'unknown')
        dataset = viewer.datasets.get(dataset_name)
        if dataset is None:
            return html.Div(f"Dataset '{dataset_name}' not found."), False

        # Get the datapoint
        if datapoint_idx >= len(dataset):
            return html.Div(f"Datapoint index {datapoint_idx} is out of range for dataset of size {len(dataset)}."), is_3d_prev

        datapoint = dataset[datapoint_idx]

        # Get is_3d from dataset info
        is_3d = dataset_info.get('is_3d', False)

        # Get class labels if available
        class_labels = dataset_info.get('class_labels', {})

        # Display the datapoint based on its type
        try:
            if is_3d:
                display = display_3d_datapoint(datapoint, point_size, point_opacity, class_labels)
            else:
                display = display_2d_datapoint(datapoint)
        except Exception as e:
            error_traceback = traceback.format_exc()
            return html.Div([
                html.H3(f"Error Loading Datapoint: {str(e)}", style={'color': 'red'}),
                html.P("Dataset type detection:"),
                html.Pre(f"Dataset class: {dataset.__class__.__name__}"),
                html.Pre(f"Is 3D: {is_3d}"),
                html.P("Datapoint structure:"),
                html.Pre(f"Inputs keys: {list(datapoint['inputs'].keys())}"),
                html.Pre(f"Labels keys: {list(datapoint['labels'].keys())}"),
                html.Pre(f"Meta info: {format_value(datapoint.get('meta_info', {}))}"),
                html.P("Error traceback:"),
                html.Pre(error_traceback, style={
                    'background-color': '#ffeeee',
                    'padding': '10px',
                    'border-radius': '5px',
                    'max-height': '300px',
                    'overflow-y': 'auto'
                })
            ]), is_3d

        return display, is_3d

    except Exception as e:
        error_traceback = traceback.format_exc()
        return html.Div([
            html.H3("Error", style={'color': 'red'}),
            html.P(f"Error loading datapoint: {str(e)}")
        ]), is_3d_prev


@callback(
    outputs=Output('view-controls', 'style'),
    inputs=[Input('is-3d-dataset', 'data')],
    group="display"
)
def update_view_controls(is_3d):
    """Update the visibility of 3D view controls based on dataset type."""
    if is_3d:
        return {'display': 'block'}
    return {'display': 'none'}
