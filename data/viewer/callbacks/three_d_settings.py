"""3D settings-related callbacks for the viewer."""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import html
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.controls.controls_3d import create_3d_controls
from data.viewer.callbacks.registry import callback


@callback(
    outputs=[
        Output('3d-settings-section', 'children'),
        Output('3d-settings-store', 'data')
    ],
    inputs=[
        Input('3d-settings-dropdown', 'value'),
        Input('3d-settings-params', 'data')
    ],
    group="3d_settings"
)
def update_3d_settings(selected_setting, setting_params):
    """Update the 3D settings section when a setting is selected or parameters change."""
    if selected_setting is None and setting_params is None:
        raise PreventUpdate

    try:
        # Get current dataset info
        dataset_info = viewer.state.get_state()['dataset_info']
        is_3d = dataset_info.get('is_3d', False)

        if not is_3d:
            return (
                html.Div("3D settings are only available for 3D datasets."),
                viewer.state.get_state()['3d_settings']
            )

        # Update state with new 3D settings
        viewer.state.update_3d_settings(selected_setting, setting_params)

        # Create updated 3D settings section
        settings_section = create_3d_settings_section()

        return (
            settings_section,
            viewer.state.get_state()['3d_settings']
        )

    except Exception as e:
        error_message = html.Div([
            html.H3("Error Updating 3D Settings", style={'color': 'red'}),
            html.P(str(e))
        ])
        return error_message, viewer.state.get_state()['3d_settings']
