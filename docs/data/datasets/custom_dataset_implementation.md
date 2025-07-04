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

class MinimalPCRDataset(BaseDataset):
    """A minimal PCR dataset example."""
    
    # Required class attributes
    SPLIT_OPTIONS = ['train', 'val', 'test']
    DATASET_SIZE = {'train': 1, 'val': 0, 'test': 0}
    INPUT_NAMES = ['src_pc', 'tgt_pc']
    LABEL_NAMES = ['transform']
    SHA1SUM = None
    
    def __init__(self, **kwargs):
        # Store any custom parameters first
        self.num_points = kwargs.pop('num_points', 100)
        # Pass remaining kwargs to base class
        super().__init__(**kwargs)
        
    def _init_annotations(self) -> None:
        """Initialize annotations for the dataset."""
        if self.split == 'train':
            self.annotations = [{'sample_id': 0}]
        else:
            self.annotations = []
    
    def _load_datapoint(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, Any]]:
        """Load a single datapoint."""
        # Create synthetic data
        src_points = torch.randn(self.num_points, 3)
        tgt_points = torch.randn(self.num_points, 3)
        transform = torch.eye(4)
        
        inputs = {
            'src_pc': {'pos': src_points},
            'tgt_pc': {'pos': tgt_points}
        }
        
        labels = {'transform': transform}
        
        # Don't add 'idx' - BaseDataset adds automatically
        meta_info = {
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
5. **Wrong constructor signature**: Must use `**kwargs` and pass to `super().__init__(**kwargs)`
6. **Adding 'idx' to meta_info**: BaseDataset adds this automatically - will cause AssertionError
7. **Incorrect DATASET_SIZE**: Must match actual annotation lengths or BaseDataset validation fails
8. **Redundant validation**: BaseDataset handles split validation, index bounds, etc.
9. **Redundant methods**: Don't override `__len__()` - BaseDataset implements as `len(self.annotations)`

## BaseDataset Automatic Features

**BaseDataset provides these automatically - don't reimplement:**

- **Index validation**: Bounds checking in `__getitem__`
- **Split validation**: Ensures split is in `SPLIT_OPTIONS`
- **Length method**: `__len__()` returns `len(self.annotations)`
- **Meta info 'idx'**: Automatically added to meta_info
- **Dataset size validation**: Compares `len(annotations)` with `DATASET_SIZE`
- **Return format**: Wraps your tuple as `{'inputs': ..., 'labels': ..., 'meta_info': ...}`

## Common Error Messages and Solutions

### `TypeError: Can't instantiate abstract class ToyCubeDataset with abstract method _init_annotations`
**Cause**: Missing `_init_annotations()` method implementation
**Solution**: Implement the required abstract method

### `AssertionError: assert type(self.split) == tuple`
**Cause**: Wrong `DATASET_SIZE` type when `self.split` is a string
**Solution**: Use `DATASET_SIZE = {'train': N, 'val': M, 'test': K}` format

### `AssertionError: Dataset class should not manually add 'idx' to meta_info`
**Cause**: Adding 'idx' key to meta_info dictionary
**Solution**: Remove 'idx' from meta_info - BaseDataset adds automatically

### `AssertionError: assert len(self) == len(self.annotations) == self.DATASET_SIZE`
**Cause**: Mismatch between actual annotation count and declared `DATASET_SIZE`
**Solution**: Ensure `DATASET_SIZE[split]` matches `len(annotations)` for each split

### Dataset returns strings instead of dictionaries
**Cause**: Dataset transforms are processing the data incorrectly
**Solution**: Check transforms configuration or set `transforms_cfg=None` for debugging

## Testing Your Dataset

Always test your dataset implementation:

```python
# Basic functionality test
dataset = YourDataset(split='train')
print(f"Dataset length: {len(dataset)}")

# Load first datapoint (BaseDataset returns dict format)
result = dataset[0]
inputs = result['inputs']
labels = result['labels'] 
meta_info = result['meta_info']
print(f"Inputs keys: {list(inputs.keys())}")
print(f"Labels keys: {list(labels.keys())}")
print(f"Meta info keys: {list(meta_info.keys())}")

# Test with data viewer
# 1. Add to data/datasets/__init__.py
# 2. Create config in configs/common/datasets/{type}/train/{name}_data_cfg.py
# 3. Add to DATASET_GROUPS in data/viewer/backend/backend.py
# 4. Run: python -m data.viewer.cli
```

## Integration with Data Viewer

**Quick checklist for data viewer integration:**

1. ✅ **Dataset class** in appropriate module (e.g., `pcr_datasets/`)
2. ✅ **Export** in `data/datasets/__init__.py` (import + `__all__`)
3. ✅ **Config file** in `configs/common/datasets/{type}/train/{name}_data_cfg.py`
4. ✅ **Register** in `DATASET_GROUPS` in `data/viewer/backend/backend.py`
5. ✅ **Test** with `python -m data.viewer.cli`

**Common data viewer issues:**
- **"init" text visible**: Add `style={'display': 'none'}` to trigger divs
- **White/blank visualization**: Check color data isn't all zeros
- **Dataset not in dropdown**: Verify all 4 integration steps above
