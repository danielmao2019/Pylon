"""Navigation-related callbacks for the viewer."""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
from data.viewer.callbacks.registry import callback


@callback(
    outputs=Output('datapoint-index-slider', 'value', allow_duplicate=True),
    inputs=[
        Input('prev-btn', 'n_clicks'),
        Input('next-btn', 'n_clicks'),
    ],
    states=[
        State('datapoint-index-slider', 'value'),
        State('datapoint-index-slider', 'min'),
        State('datapoint-index-slider', 'max'),
    ],
    group="navigation"
)
def update_index_from_buttons(prev_clicks, next_clicks, current_value, min_value, max_value):
    """Update the datapoint index based on prev/next button clicks."""
    if prev_clicks is None and next_clicks is None:
        raise PreventUpdate

    # Determine which button was clicked
    if prev_clicks is not None:
        # Previous button clicked
        new_value = max(min_value, current_value - 1)
    else:
        # Next button clicked
        new_value = min(max_value, current_value + 1)

    return new_value


@callback(
    outputs=Output('current-index-display', 'children'),
    inputs=[Input('datapoint-index-slider', 'value')],
    states=[State('dataset-info', 'data')],
    group="navigation"
)
def update_index_display(current_value, dataset_info):
    """Update the display of the current datapoint index."""
    if dataset_info is None or dataset_info == {}:
        return "No dataset loaded"

    total = dataset_info.get('length', 0)
    return f"Datapoint {current_value + 1} of {total}"
