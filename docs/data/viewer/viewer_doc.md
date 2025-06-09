# Dataset Viewer Documentation

## Overview

The Dataset Viewer is a powerful tool for visualizing and exploring datasets. It provides an interactive web interface built with Dash that allows users to:

- Browse and select datasets
- Navigate through dataset items
- Apply transforms to data
- Visualize 2D and 3D data
- Customize visualization settings

## Quick Start

```python
from data.viewer import DatasetViewer

# Create and run the viewer
viewer = DatasetViewer()
viewer.run(debug=True)
```

## Architecture

The viewer is built with a modular architecture:

```
data/viewer/
├── viewer.py              # Main viewer class
├── states/               # State management
│   ├── __init__.py
│   └── viewer_state.py   # Viewer state class
├── callbacks/            # Callback handlers
│   ├── __init__.py
│   ├── registry.py      # Callback registry
│   ├── dataset.py       # Dataset callbacks
│   ├── display.py       # Display callbacks
│   └── transforms.py    # Transform callbacks
├── layout/              # UI components
│   ├── __init__.py
│   ├── controls/        # Control components
│   └── display/         # Display components
└── managers/            # Data management
    ├── __init__.py
    └── dataset_manager.py  # Dataset manager
```

### Key Components

1. **DatasetViewer**: Main class that initializes and runs the viewer
2. **ViewerState**: Manages application state and history
3. **DatasetManager**: Handles dataset loading and caching
4. **CallbackRegistry**: Manages callback registration and dependencies
5. **UI Components**: Modular components for controls and display

## API Reference

### DatasetViewer

```python
class DatasetViewer:
    """Dataset viewer class for visualization of datasets."""

    def __init__(self, log_level: str = "INFO", log_file: Optional[str] = None):
        """Initialize the dataset viewer.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional path to log file
        """

    def run(self, debug: bool = False, host: str = "0.0.0.0", port: int = 8050) -> None:
        """Run the viewer application."""
```

### ViewerState

```python
class ViewerState:
    """Manages the viewer's state and configuration."""

    def update_dataset_info(self, name: str, length: int, class_labels: Dict[int, str],
                          is_3d: bool, available_transforms: List[str] = None) -> None:
        """Update the current dataset information."""

    def update_index(self, index: int) -> None:
        """Update the current datapoint index."""

    def update_transforms(self, transforms: Dict[str, bool]) -> None:
        """Update the transform settings."""
```

### DatasetManager

```python
class DatasetManager:
    """Manages dataset loading, caching, and operations."""

    def load_dataset(self, dataset_name: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Load a dataset."""

    def get_datapoint(self, dataset_name: str, index: int) -> Optional[Any]:
        """Get a datapoint from the dataset."""
```

## Examples

### Basic Usage

```python
from data.viewer import DatasetViewer

# Create viewer
viewer = DatasetViewer()

# Run with custom settings
viewer.run(
    debug=True,
    host="localhost",
    port=8050
)
```

### Custom Transforms

```python
from data.viewer import DatasetViewer
from data.viewer.callbacks.registry import callback

# Create viewer
viewer = DatasetViewer()

# Register custom transform
@callback(
    outputs=Output('datapoint-display', 'children'),
    inputs=[Input('transform-checkbox', 'value')],
    group='transforms'
)
def apply_custom_transform(transform_enabled):
    if transform_enabled:
        # Apply custom transform
        return transformed_data
    return original_data
```
