"""Index-related callbacks for the viewer."""
from dash import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
import html
from data.viewer.states.viewer_state import ViewerEvent
from data.viewer.layout.controls.index import create_index_controls

def register_index_callbacks(app, viewer):
    """Register callbacks related to index operations."""
    
    @app.callback(
        [
            Output('datapoint-display', 'children'),
            Output('datapoint-index', 'data'),
            Output('index-controls', 'children')
        ],
        [Input('datapoint-index-slider', 'value')],
        prevent_initial_call=True
    )
    def update_datapoint(index):
        """Update the displayed datapoint when the index changes."""
        if index is None:
            raise PreventUpdate
            
        try:
            # Update state
            viewer.state.update_index(index)
            
            # Get current dataset info
            dataset_info = viewer.state.get_state()['dataset_info']
            dataset_name = dataset_info['name']
            
            # Get dataset
            dataset = viewer.datasets.get(dataset_name)
            if dataset is None:
                return (
                    html.Div("Error: Dataset not found."),
                    index,
                    create_index_controls(dataset_info)
                )
            
            # Get datapoint
            datapoint = dataset[index]
            
            # Create display content
            display_content = html.Div([
                html.H3(f"Datapoint {index}"),
                html.Div([
                    html.Strong("Class: "),
                    html.Span(dataset_info['class_labels'].get(datapoint['label'], 'Unknown'))
                ]),
                html.Div([
                    html.Strong("Image Shape: "),
                    html.Span(str(datapoint['image'].shape))
                ])
            ])
            
            return (
                display_content,
                index,
                create_index_controls(dataset_info)
            )
            
        except Exception as e:
            error_message = html.Div([
                html.H3("Error Loading Datapoint", style={'color': 'red'}),
                html.P(str(e))
            ])
            return error_message, index, create_index_controls(viewer.state.get_state()['dataset_info']) 