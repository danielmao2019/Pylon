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

## Dataset Configuration

The viewer automatically discovers datasets from config files in specific directories. To add a new dataset:

### 1. Create Dataset Class

Implement your dataset class inheriting from `BaseDataset` (see [Custom Dataset Implementation Guide](../datasets/custom_dataset_implementation.md)).

### 2. Add Dataset to Module

Add your dataset to `data/datasets/__init__.py`:

```python
# Add import
from data.datasets.your_module.your_dataset import YourDataset

# Add to __all__
__all__ = (
    # ... existing datasets
    'YourDataset',
)
```

### 3. Create Config File

Create a config file in the appropriate directory:

- **Semantic Segmentation**: `configs/common/datasets/semantic_segmentation/train/your_dataset_data_cfg.py`
- **2D Change Detection**: `configs/common/datasets/change_detection/train/your_dataset_data_cfg.py`
- **3D Change Detection**: `configs/common/datasets/change_detection/train/your_dataset_data_cfg.py`
- **Point Cloud Registration**: `configs/common/datasets/point_cloud_registration/train/your_dataset_data_cfg.py`

Config file format:
```python
import data

data_cfg = {
    'train_dataset': {
        'class': data.datasets.YourDataset,
        'args': {
            'split': 'train',
            'param1': 'value1',
            'param2': 'value2',
        },
    },
}
```

### 4. Register Dataset Type

Add your dataset to the appropriate group in `data/viewer/backend/backend.py`:

```python
DATASET_GROUPS = {
    'semseg': ['coco_stuff_164k'],
    '2dcd': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
    '3dcd': ['urb3dcd', 'slpccd'],
    'pcr': ['synth_pcr', 'real_pcr', 'kitti', 'your_dataset'],  # Add here
}
```

### Example: ToyCubeDataset

Here's how the `ToyCubeDataset` was added:

1. **Created dataset class** in `data/datasets/pcr_datasets/toy_cube_dataset.py`
2. **Added to module** in `data/datasets/__init__.py`
3. **Created config file** as `configs/common/datasets/point_cloud_registration/train/toy_cube_data_cfg.py`
4. **Registered in backend** by adding `'toy_cube'` to the `'pcr'` group

The viewer will automatically discover and load the dataset, making it available in the UI dropdown.

## Troubleshooting

### Common Issues

1. **"Can't instantiate abstract class"**
   - Make sure your dataset implements `_init_annotations()` method
   - Call `super().__init__()` after setting instance variables

2. **Dataset not appearing in viewer**
   - Check config file is in the correct directory
   - Verify dataset is added to `DATASET_GROUPS` in backend
   - Ensure dataset is properly exported from `data.datasets`

3. **Config file not found**
   - Config filename must match pattern: `{dataset_name}_data_cfg.py`
   - Place in the correct subdirectory based on dataset type

4. **Import errors**
   - Make sure all dependencies are available
   - Check that dataset class is properly imported in `__init__.py`
