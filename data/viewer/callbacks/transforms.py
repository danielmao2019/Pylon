"""Transform-related callbacks for the viewer."""
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import html
import traceback
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.controls.transforms import create_transforms_section


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

        # Get list of selected transforms
        selected_transforms = [
            transform for transform, selected in zip(dataset_info['available_transforms'], transform_values)
            if selected
        ]

        # Apply transforms using dataset manager
        datapoint = viewer.dataset_manager.apply_transforms(dataset_name, datapoint_idx, selected_transforms)
        if datapoint is None:
            return html.Div([
                html.H3("Error Applying Transforms", style={'color': 'red'}),
                html.P("Failed to apply transforms to datapoint.")
            ])

        # Display the transformed datapoint
        try:
            if dataset_info['is_3d']:
                display = display_3d_datapoint(datapoint, class_labels=dataset_info['class_labels'])
            else:
                display = display_2d_datapoint(datapoint)
            return display
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

    @app.callback(
        [
            Output('transforms-section', 'children'),
            Output('transforms-store', 'data')
        ],
        [
            Input('transforms-dropdown', 'value'),
            Input('transforms-params', 'data')
        ],
        prevent_initial_call=True
    )
    def update_transforms(selected_transform, transform_params):
        """Update the transforms section when a transform is selected or parameters change."""
        if selected_transform is None and transform_params is None:
            raise PreventUpdate

        try:
            # Get current dataset info
            dataset_info = viewer.state.get_state()['dataset_info']
            available_transforms = dataset_info.get('available_transforms', [])

            # Update state with new transform
            viewer.state.update_transforms(selected_transform, transform_params)

            # Create updated transforms section
            transforms_section = create_transforms_section(available_transforms)

            return (
                transforms_section,
                viewer.state.get_state()['transforms']
            )

        except Exception as e:
            error_message = html.Div([
                html.H3("Error Updating Transforms", style={'color': 'red'}),
                html.P(str(e))
            ])
            return error_message, viewer.state.get_state()['transforms']
