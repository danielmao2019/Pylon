"""Transform-related callbacks for the viewer."""
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import html
import traceback
from data.viewer.utils.dataset_utils import is_3d_dataset
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint

def register_transform_callbacks(app, viewer):
    """Register callbacks related to transform operations."""
    
    @app.callback(
        Output('datapoint-display', 'children', allow_duplicate=True),
        [Input({'type': 'transform-checkbox', 'index': ALL}, 'value')],
        [
            State('dataset-info', 'data'),
            State('datapoint-index-slider', 'value')
        ],
        prevent_initial_call=True
    )
    def apply_transforms(transform_values, dataset_info, datapoint_idx):
        """Apply the selected transforms to the current datapoint."""
        if not dataset_info or 'name' not in dataset_info:
            raise PreventUpdate

        dataset_name = dataset_info['name']
        dataset = viewer.datasets.get(dataset_name)

        if dataset is None or datapoint_idx >= len(dataset):
            raise PreventUpdate

        try:
            # Get the datapoint
            datapoint = dataset[datapoint_idx]

            # Determine if this is a 3D dataset
            is_3d = is_3d_dataset(dataset, datapoint)

            # Get class labels if available
            class_labels = dataset_info.get('class_labels', {})

            # Apply transforms if any are selected
            if transform_values and any(transform_values):
                # Get the list of selected transforms
                selected_transforms = [
                    transform for transform, selected in zip(viewer.available_transforms, transform_values)
                    if selected
                ]

                # Apply each selected transform
                for transform in selected_transforms:
                    if transform in viewer.transform_functions:
                        datapoint = viewer.transform_functions[transform](datapoint)

            # Display the transformed datapoint
            try:
                if is_3d:
                    display = display_3d_datapoint(datapoint, class_labels=class_labels)
                else:
                    display = display_2d_datapoint(datapoint)
            except Exception as e:
                error_traceback = traceback.format_exc()
                return html.Div([
                    html.H3(f"Error Displaying Transformed Datapoint: {str(e)}", style={'color': 'red'}),
                    html.P("Error traceback:"),
                    html.Pre(error_traceback, style={
                        'background-color': '#ffeeee',
                        'padding': '10px',
                        'border-radius': '5px',
                        'max-height': '300px',
                        'overflow-y': 'auto'
                    })
                ])

            return display

        except Exception as e:
            error_traceback = traceback.format_exc()
            return html.Div([
                html.H3(f"Error Applying Transforms: {str(e)}", style={'color': 'red'}),
                html.P("Error traceback:"),
                html.Pre(error_traceback, style={
                    'background-color': '#ffeeee',
                    'padding': '10px',
                    'border-radius': '5px',
                    'max-height': '300px',
                    'overflow-y': 'auto'
                })
            ]) 