# Custom Dataset Implementation Guide

This guide explains how to implement custom datasets in Pylon, based on lessons learned from creating the `ToyCubeDataset`.

## BaseDataset Abstract Class Requirements

All custom datasets must inherit from `BaseDataset` and implement the following abstract methods:

### Required Methods

1. **`_init_annotations(self) -> None`**
   - Called during `__init__()` after `super().__init__()`
   - Must populate `self.annotations` list with annotation dictionaries
   - Each annotation represents one data sample
   - Example:
   ```python
   def _init_annotations(self) -> None:
       """Initialize annotations for the dataset."""
       if self.split == 'train':
           self.annotations = [{
               'idx': 0,
               'src_cube_id': 'source_cube',
               'tgt_cube_id': 'target_cube',
               'transform_id': 'rotation_translation'
           }]
       else:
           self.annotations = []
   ```

2. **`_load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]`**
   - Loads actual data for a given index
   - Returns tuple of (inputs, labels, meta_info)
   - `inputs`: Dictionary containing input tensors
   - `labels`: Dictionary containing ground truth labels
   - `meta_info`: Dictionary containing metadata
   - Example:
   ```python
   def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
       # Load your data here
       inputs = {'src_pc': {'pos': points, 'rgb': colors}}
       labels = {'transform': transform_matrix}
       meta_info = {'idx': idx, 'description': 'sample info'}
       return inputs, labels, meta_info
   ```

### Optional Methods

- **`__len__(self) -> int`**: If not implemented, defaults to `len(self.annotations)`

## Common Implementation Patterns

### 1. Constructor Pattern
```python
def __init__(self, split: str = 'train', **kwargs):
    # Store parameters first
    self.split = split
    self.param1 = kwargs.get('param1', default_value)
    
    # Call parent constructor (this calls _init_annotations)
    super().__init__()
```

### 2. Annotation Pattern
```python
def _init_annotations(self) -> None:
    """Initialize annotations based on split and parameters."""
    if self.split == 'train':
        # Load training annotations
        self.annotations = self._load_training_annotations()
    elif self.split == 'val':
        # Load validation annotations
        self.annotations = self._load_validation_annotations()
    else:
        # Empty or test annotations
        self.annotations = []
```

### 3. Data Loading Pattern
```python
def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
    """Load datapoint using annotation metadata."""
    annotation = self.annotations[idx]
    
    # Use annotation metadata to load actual data
    data = self._load_from_annotation(annotation)
    
    # Format according to task requirements
    inputs = self._format_inputs(data)
    labels = self._format_labels(data)
    meta_info = self._format_meta_info(annotation, data)
    
    return inputs, labels, meta_info
```

## Task-Specific Format Requirements

### Point Cloud Registration (PCR)
- **Inputs**: `{'src_pc': {'pos': tensor, 'rgb': tensor}, 'tgt_pc': {'pos': tensor, 'rgb': tensor}}`
- **Labels**: `{'transform': 4x4_matrix}`
- **Meta Info**: Should include point counts, IDs, descriptions

### Change Detection (2D)
- **Inputs**: `{'image': {'img_1': tensor, 'img_2': tensor}}`
- **Labels**: `{'change_map': tensor}`

### Semantic Segmentation
- **Inputs**: `{'image': {'image': tensor}}`
- **Labels**: `{'label': tensor}`

## Example: ToyCubeDataset

Here's a complete example of a minimal PCR dataset:

```python
from typing import Tuple, Dict, Any
import torch
from data.datasets.base_dataset import BaseDataset

class ToyCubeDataset(BaseDataset):
    """A toy PCR dataset for testing."""
    
    def __init__(self, split: str = 'train', cube_density: int = 8):
        self.split = split
        self.cube_density = cube_density
        super().__init__()
        
    def _init_annotations(self) -> None:
        """Initialize annotations for the dataset."""
        if self.split == 'train':
            self.annotations = [{
                'idx': 0,
                'src_cube_id': 'source_cube',
                'tgt_cube_id': 'target_cube',
                'transform_id': 'rotation_translation'
            }]
        else:
            self.annotations = []
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a single datapoint."""
        # Create synthetic data
        src_points = torch.randn(100, 3)
        tgt_points = torch.randn(100, 3)
        transform = torch.eye(4)
        
        inputs = {
            'src_pc': {'pos': src_points},
            'tgt_pc': {'pos': tgt_points}
        }
        
        labels = {'transform': transform}
        
        meta_info = {
            'idx': idx,
            'src_points_count': len(src_points),
            'tgt_points_count': len(tgt_points)
        }
        
        return inputs, labels, meta_info
```

## Common Pitfalls

1. **Forgetting `_init_annotations`**: Will result in `TypeError: Can't instantiate abstract class`
2. **Wrong constructor order**: Call `super().__init__()` AFTER setting instance variables
3. **Empty annotations**: Make sure `self.annotations` is properly populated
4. **Incorrect return format**: `_load_datapoint` must return exactly 3 dictionaries

## Testing Your Dataset

Always test your dataset implementation:

```python
# Basic functionality test
dataset = YourDataset(split='train')
print(f"Dataset length: {len(dataset)}")

# Load first datapoint
inputs, labels, meta_info = dataset[0]
print(f"Inputs keys: {inputs.keys()}")
print(f"Labels keys: {labels.keys()}")
print(f"Meta info: {meta_info}")
```
