"""Navigation-related callbacks for the viewer."""
from typing import List, Optional
from dash import Input, Output, State, html, callback_context
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
def update_index_from_buttons(
    prev_clicks: Optional[int],
    next_clicks: Optional[int],
    current_value: int,
    min_value: int,
    max_value: int
) -> List[int]:
    """Update the datapoint index based on prev/next button clicks."""
    if prev_clicks is None and next_clicks is None:
        raise PreventUpdate
    assert isinstance(current_value, int)

    # Get the ID of the button that triggered the callback
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    # Update value based on which button was clicked
    if triggered_id == 'prev-btn':
        new_value = max(min_value, current_value - 1)
    else:  # next-btn
        new_value = min(max_value, current_value + 1)

    return [new_value]


@callback(
    outputs=Output('current-index-display', 'children'),
    inputs=[Input('datapoint-index-slider', 'value')],
    group="navigation"
)
def update_current_index(current_idx: int) -> List[str]:
    """Update the current index display."""
    assert isinstance(current_idx, int)
    return [f"Current Index: {current_idx}"]
