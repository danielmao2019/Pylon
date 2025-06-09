"""UI components for displaying dataset items."""
from dash import html


def create_dataset_info_display(dataset_info=None):
    """
    Create a display of dataset information.
    
    Args:
        dataset_info: Dictionary containing dataset information
        
    Returns:
        html.Div containing dataset info
    """
    if dataset_info is None or not dataset_info:
        return html.Div([
            html.H3("Dataset Information"),
            html.P("No dataset selected.")
        ])
    
    # Extract basic dataset info
    dataset_name = dataset_info.get('name', 'Unknown')
    dataset_length = dataset_info.get('length', 0)
    class_labels = dataset_info.get('class_labels', {})
    
    # Create class labels display if available
    class_labels_display = []
    if class_labels:
        class_labels_display = [
            html.H4("Class Labels:"),
            html.Ul([
                html.Li(f"{idx}: {name}") for idx, name in class_labels.items()
            ], style={'max-height': '150px', 'overflow-y': 'auto'})
        ]
    
    return html.Div([
        html.H3("Dataset Information"),
        html.Ul([
            html.Li(f"Name: {dataset_name}"),
            html.Li(f"Number of items: {dataset_length}")
        ]),
        html.Div(class_labels_display)
    ])
