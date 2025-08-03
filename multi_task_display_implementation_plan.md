# Multi-Task Learning Datapoint Display Implementation Plan

## Overview

This document outlines the implementation plan for multi-task learning datapoint display functionality in Pylon. The goal is to enable composable visualization of multi-task datasets like NYUv2 (RGB + depth + normals + segmentation + edges) by treating `data/viewer/` as a utility module that provides atomic display functions for implementing `display_datapoint` methods in dataset classes.

## Design Philosophy Alignment

Following Pylon's core design principles from CLAUDE.md:

### ⚠️ CRITICAL: Fail-Fast Philosophy
- **NO defensive programming** - let code crash with clear error messages
- **Use assertions extensively** for input validation
- **Always use kwargs** for function calls to prevent parameter ordering bugs
- **Investigate root causes** rather than handling symptoms

### Framework Integration
- **Simple composition**: Each dataset imports needed utilities and composes its own display
- **Explicit over implicit**: No auto-detection, no magic - datasets specify exactly what they need
- **Maintain backward compatibility**: Existing single-task displays continue unchanged
- **Performance-conscious**: Utilize existing `ParallelFigureCreator` and LOD optimizations

## Current State Analysis

### Existing Display Architecture
1. **Task-specific displays**: `display_semseg.py`, `display_pcr.py`, `display_2dcd.py`, `display_3dcd.py`
2. **Base dataset method**: `display_datapoint()` returns `None` by default (fallback)
3. **Task-specific overrides**: `BaseSemsegDataset.display_datapoint()` provides specialized display
4. **Utility functions**: `data/viewer/utils/` contains modular display components
5. **Multi-task gap**: Multi-task datasets inherit from `BaseDataset` directly, lack display methods

### Multi-Task Datasets
- **NYUv2Dataset**: RGB + depth + normals + semantic segmentation + edge detection
- **ADE20KDataset**: RGB + semantic segmentation + instance segmentation + object parts
- **PASCALContextDataset**: RGB + semantic segmentation + human parts + object detection
- **CityScapesDataset**: RGB + semantic segmentation + instance segmentation + depth

## Simplified Implementation Strategy

### Create Atomic Display Utilities

Create simple utility functions in `data/viewer/utils/atomic_displays/`:

```
data/viewer/utils/atomic_displays/
├── __init__.py
├── image_display.py         # RGB/grayscale image visualization (consolidate existing)
├── depth_display.py         # Depth map visualization  
├── normal_display.py        # Surface normal visualization
├── edge_display.py          # Edge detection visualization
├── segmentation_display.py  # Semantic/instance segmentation (move existing)
├── point_cloud_display.py   # Point cloud visualization (move existing)
└── detection_display.py     # Object detection boxes/masks
```

**Key Design Principles:**
- **Simple utility functions**: Each function does one thing well
- **Fail-fast validation**: Each component validates inputs with assertions
- **Type safety**: Full type annotations for all functions
- **Explicit imports**: Datasets import exactly what they need
- **Performance**: Leverage existing optimizations (LOD, parallel creation)

**Example atomic display signature:**
```python
def create_image_display(
    image: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create image display with proper validation."""
    # CRITICAL: Input validation with assertions
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    assert image.ndim == 3, f"Expected 3D tensor [C,H,W], got shape {image.shape}"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Use existing create_image_figure implementation
    return create_image_figure(image=image, title=title)
```

### Dataset Integration Examples

Each multi-task dataset imports utilities and composes its own display:

```python
# NYUv2Dataset example
from dash import html, dcc
from data.viewer.utils.atomic_displays import (
    create_image_display,
    create_depth_display, 
    create_normal_display,
    create_segmentation_display,
    create_edge_display
)
from data.viewer.utils.display_utils import (
    DisplayStyles,
    ParallelFigureCreator,
    create_figure_grid,
    create_standard_datapoint_layout
)

class NYUv2Dataset(BaseDataset):
    @staticmethod
    def display_datapoint(
        datapoint: Dict[str, Any],
        class_labels: Optional[Dict[str, List[str]]] = None,
        camera_state: Optional[Dict[str, Any]] = None,
        settings_3d: Optional[Dict[str, Any]] = None
    ) -> html.Div:
        """Display NYUv2 multi-task datapoint."""
        
        # Input validation
        assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
        assert 'inputs' in datapoint, f"datapoint missing 'inputs', got keys: {list(datapoint.keys())}"
        assert 'labels' in datapoint, f"datapoint missing 'labels', got keys: {list(datapoint.keys())}"
        
        # Extract data
        inputs = datapoint['inputs']
        labels = datapoint['labels']
        
        # Create figure tasks for parallel execution
        figure_tasks = [
            lambda: create_image_display(
                image=inputs['image'],
                title="RGB Image"
            ),
            lambda: create_depth_display(
                depth=labels['depth_estimation'],
                title="Depth Map"
            ),
            lambda: create_normal_display(
                normals=labels['normal_estimation'],
                title="Surface Normals"
            ),
            lambda: create_segmentation_display(
                segmentation=labels['semantic_segmentation'],
                title="Semantic Segmentation",
                class_labels=class_labels
            ),
            lambda: create_edge_display(
                edges=labels['edge_detection'],
                title="Edge Detection"
            )
        ]
        
        # Create figures in parallel
        figure_creator = ParallelFigureCreator(max_workers=5)
        figures = figure_creator.create_figures_parallel(figure_tasks)
        
        # Create grid layout (2x3 for 5 figures)
        figure_components = create_figure_grid(
            figures=figures,
            width_style="33%",
            height_style="400px"
        )
        
        # Use standard layout with statistics
        return create_standard_datapoint_layout(
            figure_components=figure_components,
            stats_components=[],  # Could add stats for each modality
            meta_info=datapoint.get('meta_info', {}),
            debug_outputs=datapoint.get('debug')
        )
```

## Atomic Display Utility Functions

Each utility function follows this simple pattern:

### Image Display
```python
def create_image_display(
    image: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create image display (RGB/grayscale)."""
    # Input validation with assertions
    assert isinstance(image, torch.Tensor), f"Expected torch.Tensor, got {type(image)}"
    assert image.ndim == 3, f"Expected 3D tensor [C,H,W], got shape {image.shape}"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Use existing create_image_figure implementation
    return create_image_figure(image=image, title=title)
```

### Depth Display
```python
def create_depth_display(
    depth: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create depth map display."""
    assert isinstance(depth, torch.Tensor), f"Expected torch.Tensor, got {type(depth)}"
    assert depth.ndim == 2, f"Expected 2D tensor [H,W], got shape {depth.shape}"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Implementation for depth visualization (colormap, etc.)
    # ... 
```

### Normal Display
```python
def create_normal_display(
    normals: torch.Tensor,
    title: str,
    **kwargs: Any
) -> go.Figure:
    """Create surface normal display."""
    assert isinstance(normals, torch.Tensor), f"Expected torch.Tensor, got {type(normals)}"
    assert normals.ndim == 3, f"Expected 3D tensor [3,H,W], got shape {normals.shape}"
    assert normals.shape[0] == 3, f"Expected 3 channels for normals, got {normals.shape[0]}"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Implementation for normal visualization
    # ...
```

### Consolidation Strategy

1. **Move existing functions**: Move point cloud and segmentation display functions to atomic_displays
2. **Consolidate image functions**: Merge create_rgb_display and create_image_figure 
3. **Simple wrappers**: Each atomic function is a simple wrapper with validation
4. **Consistent API**: All functions follow same signature pattern (data, title, **kwargs)

### Performance Optimizations

1. **Parallel Figure Creation**: Use existing `ParallelFigureCreator` in dataset implementations
2. **LOD Integration**: Existing LOD system continues to work for point cloud displays
3. **Memory Management**: Datasets control their own memory usage patterns
4. **Existing Utilities**: Leverage all existing display_utils functions

## Implementation Phases

### Phase 1: Create Atomic Display Structure
- [ ] Create `data/viewer/utils/atomic_displays/` directory
- [ ] Move existing point cloud display functions from utils/
- [ ] Move existing segmentation display functions from utils/
- [ ] Consolidate image display functions (remove redundancy)

### Phase 2: Implement New Atomic Displays
- [ ] Create depth display utility function
- [ ] Create normal display utility function  
- [ ] Create edge display utility function
- [ ] Add comprehensive input validation with assertions to all functions

### Phase 3: Dataset Integration
- [ ] Implement NYUv2Dataset.display_datapoint() using atomic utilities
- [ ] Test integration with data viewer
- [ ] Implement other multi-task datasets (ADE20K, CityScapes, etc.)
- [ ] Verify all displays work correctly

### Phase 4: Testing & Cleanup
- [ ] Create unit tests for each atomic display function
- [ ] Integration tests with real multi-task datasets
- [ ] Performance testing (ensure no regression)
- [ ] Clean up any unused legacy functions

## Error Handling Strategy

Following Pylon's fail-fast philosophy:

1. **Assertions in atomic functions**: Every atomic display validates inputs with assertions
2. **Clear error messages**: All assertion messages include actual vs expected values
3. **No defensive programming**: Let invalid states crash with clear messages
4. **Dataset-level validation**: Each dataset validates its own datapoint structure

Example atomic function validation:
```python
def create_depth_display(depth: torch.Tensor, title: str, **kwargs: Any) -> go.Figure:
    """Create depth map display with fail-fast validation."""
    assert isinstance(depth, torch.Tensor), f"Expected torch.Tensor, got {type(depth)}"
    assert depth.ndim == 2, f"Expected 2D tensor [H,W], got shape {depth.shape}"
    assert depth.dtype == torch.float32, f"Expected float32, got {depth.dtype}"
    assert depth.numel() > 0, f"Depth tensor cannot be empty"
    assert isinstance(title, str), f"Expected str title, got {type(title)}"
    
    # Implementation continues...
```

Example dataset validation:
```python
def display_datapoint(self, datapoint: Dict[str, Any], **kwargs) -> html.Div:
    """NYUv2 display with dataset-specific validation."""
    # Validate datapoint structure
    assert isinstance(datapoint, dict), f"datapoint must be dict, got {type(datapoint)}"
    assert 'inputs' in datapoint, f"datapoint missing 'inputs', got keys: {list(datapoint.keys())}"
    assert 'labels' in datapoint, f"datapoint missing 'labels', got keys: {list(datapoint.keys())}"
    
    # Validate expected keys for NYUv2
    inputs = datapoint['inputs']
    labels = datapoint['labels']
    assert 'image' in inputs, f"inputs missing 'image', got keys: {list(inputs.keys())}"
    assert 'depth_estimation' in labels, f"labels missing 'depth_estimation', got keys: {list(labels.keys())}"
    # ... validate other expected keys
```

## Testing Strategy

Following Pylon testing guidelines:

1. **No pytest.skip()**: Auto-generate test data if needed
2. **Pytest functions only**: No test classes
3. **Atomic function tests**: Test each display utility function independently
4. **Input validation tests**: Test all assertion conditions for atomic functions
5. **Integration tests**: Test complete dataset display_datapoint implementations
6. **Performance tests**: Ensure no regression compared to existing displays

Example test structure:
```python
def test_create_depth_display_valid_input():
    """Test depth display with valid input."""
    depth = torch.randn(480, 640, dtype=torch.float32)
    title = "Test Depth"
    
    figure = create_depth_display(depth=depth, title=title)
    
    assert figure is not None
    assert figure.layout.title.text == title

def test_create_depth_display_invalid_shape():
    """Test depth display fails with wrong shape."""
    depth = torch.randn(3, 480, 640)  # Wrong: should be 2D
    
    with pytest.raises(AssertionError, match="Expected 2D tensor"):
        create_depth_display(depth=depth, title="Test")
```

## Success Criteria

1. **Backward compatibility**: Existing single-task displays work unchanged
2. **Performance**: Multi-task displays perform at least as well as existing displays
3. **Simplicity**: Easy for datasets to import and use atomic functions
4. **Maintainability**: Clear separation of atomic utilities and dataset implementations
5. **User experience**: Rich multi-task visualizations in data viewer

## Benefits of Simplified Approach

1. **No magic**: Datasets explicitly specify what they need
2. **Easy debugging**: Clear import and function call chains
3. **Flexible layouts**: Each dataset controls its own layout
4. **Reusable utilities**: Atomic functions can be used by any dataset
5. **Performance control**: Datasets control parallel execution and memory usage

## Implementation Status

- [x] **Design complete**: Simplified approach with atomic utilities
- [ ] **Ready to implement**: Phase 1 (create atomic display structure)

This simplified plan removes complexity while providing all the functionality needed for multi-task dataset visualization. Each dataset has full control over its display logic while leveraging reusable atomic utilities.