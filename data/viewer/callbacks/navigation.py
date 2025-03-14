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
    assert isinstance(current_value, int)

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
    group="navigation"
)
def update_current_index(current_idx):
    """Update the current index display."""
    assert isinstance(current_idx, int)
    return [f"Current Index: {current_idx}"]
