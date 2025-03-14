"""Display-related callbacks for the viewer."""
from dash import Input, Output, State, html
import logging
from data.viewer.layout.display.display_2d import display_2d_datapoint
from data.viewer.layout.display.display_3d import display_3d_datapoint
from data.viewer.callbacks.registry import callback, registry

logger = logging.getLogger(__name__)

@callback(
    outputs=[
        Output('datapoint-display', 'children'),
    ],
    inputs=[
        Input('dataset-info', 'data'),
        Input('datapoint-index-slider', 'value'),
        Input('point-size-slider', 'value'),
        Input('point-opacity-slider', 'value')
    ],
    group="display"
)
def update_datapoint(dataset_info, datapoint_idx, point_size, point_opacity):
    """
    Update the displayed datapoint based on the slider value.
    Also handles 3D point cloud visualization settings.
    """
    logger.info(f"Display update callback triggered - Dataset info: {dataset_info}, Index: {datapoint_idx}")
    
    if dataset_info is None or dataset_info == {}:
        logger.warning("No dataset info available")
        return html.Div("No dataset loaded.")

    dataset_name = dataset_info.get('name', 'unknown')
    logger.info(f"Attempting to get dataset: {dataset_name}")
    
    # Get dataset from manager through registry
    dataset = registry.viewer.dataset_manager.get_dataset(dataset_name)
    if dataset is None:
        logger.error(f"Dataset '{dataset_name}' not found in manager")
        return html.Div(f"Dataset '{dataset_name}' not found.")

    # Get the datapoint
    if datapoint_idx >= len(dataset):
        logger.error(f"Datapoint index {datapoint_idx} out of range for dataset of size {len(dataset)}")
        return html.Div(f"Datapoint index {datapoint_idx} is out of range for dataset of size {len(dataset)}.")

    logger.info(f"Loading datapoint at index {datapoint_idx}")
    datapoint = dataset[datapoint_idx]

    # Get is_3d from dataset info
    is_3d = dataset_info.get('is_3d', False)
    logger.info(f"Dataset type: {'3D' if is_3d else '2D'}")

    # Get class labels if available
    class_labels = dataset_info.get('class_labels', {})
    logger.info(f"Class labels available: {bool(class_labels)}")

    # Display the datapoint based on its type
    if is_3d:
        logger.info("Creating 3D display")
        display = display_3d_datapoint(datapoint, point_size, point_opacity, class_labels)
    else:
        logger.info("Creating 2D display")
        display = display_2d_datapoint(datapoint)
    logger.info("Display created successfully")
    return display
