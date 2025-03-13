"""Navigation-related callbacks for the viewer."""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate

def register_navigation_callbacks(app, viewer):
    """Register callbacks related to navigation operations."""
    
    @app.callback(
        Output('datapoint-index-slider', 'value'),
        [
            Input('prev-btn', 'n_clicks'),
            Input('next-btn', 'n_clicks'),
        ],
        [
            State('datapoint-index-slider', 'value'),
            State('datapoint-index-slider', 'min'),
            State('datapoint-index-slider', 'max'),
        ],
        prevent_initial_call=True
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

    @app.callback(
        Output('current-index-display', 'children'),
        [Input('datapoint-index-slider', 'value')],
        [State('dataset-info', 'data')]
    )
    def update_index_display(current_value, dataset_info):
        """Update the display of the current datapoint index."""
        if dataset_info is None or dataset_info == {}:
            return "No dataset loaded"
            
        total = dataset_info.get('length', 0)
        return f"Datapoint {current_value + 1} of {total}" 