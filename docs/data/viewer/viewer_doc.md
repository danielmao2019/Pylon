# Dataset Viewer Documentation

## Overview

The Dataset Viewer is a powerful tool for visualizing and exploring various types of computer vision datasets. It provides an interactive web interface built with Dash that allows users to:

- Browse and select datasets
- Navigate through dataset items
- Apply transforms to data
- Visualize different types of data:
  - 2D Change Detection (before/after images with change maps)
  - 3D Change Detection (point cloud pairs with change maps)
  - Point Cloud Registration (source/target point clouds with transformations)
  - Semantic Segmentation (images with segmentation masks)
- Customize visualization settings
- View dataset statistics and metadata

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
├── cli.py                # Command-line interface
├── states/               # State management
│   ├── __init__.py
│   └── viewer_state.py   # Viewer state class
├── callbacks/            # Callback handlers
│   ├── __init__.py
│   ├── registry.py      # Callback registry
│   ├── dataset.py       # Dataset callbacks
│   ├── display.py       # Display callbacks
│   ├── navigation.py    # Navigation callbacks
│   ├── transforms.py    # Transform callbacks
│   └── three_d_settings.py  # 3D visualization settings
├── layout/              # UI components
│   ├── __init__.py
│   ├── app.py          # Main app layout
│   ├── controls/       # Control components
│   └── display/        # Display components
└── managers/           # Data management
    ├── __init__.py
    ├── dataset_manager.py  # Dataset manager
    ├── dataset_cache.py    # Caching system
    ├── transform_manager.py # Transform management
    └── registry.py         # Dataset type registry
```

### Key Components

1. **DatasetViewer**: Main class that initializes and runs the viewer
2. **ViewerState**: Manages application state, history, and events
3. **DatasetManager**: Handles dataset loading, caching, and operations
4. **CallbackRegistry**: Manages callback registration and dependencies
5. **UI Components**: Modular components for controls and display
6. **TransformManager**: Handles data transformations and preprocessing

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
                          transforms: List[Dict[str, Any]] = None, dataset_type: str = None) -> None:
        """Update the current dataset information."""

    def update_index(self, index: int) -> None:
        """Update the current datapoint index."""

    def update_transforms(self, transforms: List[Dict[str, Any]]) -> None:
        """Update the transform settings."""

    def update_3d_settings(self, point_size: float, point_opacity: float) -> None:
        """Update 3D visualization settings."""
```

### DatasetManager

```python
class DatasetManager:
    """Manages dataset loading, caching, and operations."""

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load a dataset and return its information."""

    def get_datapoint(self, dataset_name: str, index: int, transform_indices: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
        """Get a datapoint with optional transforms applied."""
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

### Command Line Usage

```bash
# Run the viewer from command line
python -m data.viewer.cli --debug --host localhost --port 8050
```

### Dataset Types

The viewer supports several dataset types:

1. **2D Change Detection (2dcd)**
   - Before/after images
   - Change maps
   - Examples: Air Change, CDD, LEVIR-CD, OSCD, SYSU-CD

2. **3D Change Detection (3dcd)**
   - Point cloud pairs
   - Change maps
   - Examples: Urb3DCD, SLPCCD

3. **Point Cloud Registration (pcr)**
   - Source/target point clouds
   - Transformations
   - Examples: Synth PCR, Real PCR, KITTI

4. **Semantic Segmentation (semseg)**
   - Images with segmentation masks
   - Class labels
   - Example: COCO Stuff 164K
