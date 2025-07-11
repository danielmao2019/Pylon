# Dataset Classes Design in Pylon

## Overview

This document explains the architecture and design patterns for dataset classes in the Pylon framework, including base class structure, inheritance patterns, and validation requirements.

## BaseDataset Architecture

### Required Class Attributes

Every dataset class must define these class-level attributes:

```python
class YourDataset(BaseDataset):
    SPLIT_OPTIONS = ['train', 'val', 'test']  # Supported splits
    INPUT_NAMES = ['input1', 'input2']        # Input tensor names (used by framework)
    LABEL_NAMES = ['label1']                  # Label tensor names (used by framework)
    DATASET_SIZE = {                         # CRITICAL: Must match actual filtered sizes
        'train': 12345,
        'val': 678, 
        'test': 910
    }
    SHA1SUM = None                           # Optional data integrity check
```

### Initialization Pattern

```python
def __init__(self, custom_param=default_value, **kwargs):
    # 1. Process custom parameters first
    self.custom_param = custom_param
    self.another_param = kwargs.pop('another_param', default)
    
    # 2. ALWAYS call super().__init__(**kwargs) last
    super().__init__(**kwargs)
```

**Why this order matters**: BaseDataset.__init__() calls _init_annotations(), which may need your custom parameters.

### Core Methods

Every dataset must implement these two methods:

```python
def _init_annotations(self):
    """Load metadata and create self.annotations list."""
    # Load metadata from files
    # Apply filtering based on split, custom parameters, etc.
    # Create self.annotations as list of dictionaries
    self.annotations = [...]

def _load_datapoint(self, idx: int) -> Tuple[Dict, Dict, Dict]:
    """Load a single datapoint."""
    annotation = self.annotations[idx]
    
    # Load actual data (images, point clouds, etc.)
    # Process and transform data
    # Return three dictionaries
    return inputs, labels, meta_info
```

## Three-Dictionary Return Pattern

### Mandatory Structure

`_load_datapoint()` must return exactly three dictionaries:

```python
# 1. INPUTS - Data fed to models
inputs = {
    'input_name1': tensor_or_dict,  # Must match INPUT_NAMES
    'input_name2': tensor_or_dict,
    # ... other inputs
}

# 2. LABELS - Ground truth for supervision  
labels = {
    'label_name1': tensor,          # Must match LABEL_NAMES
    # ... other labels
}

# 3. META_INFO - Metadata for analysis/debugging
meta_info = {
    'idx': idx,                     # Added automatically by BaseDataset
    'file_path': str,               # Example metadata
    'scene_name': str,              # Example metadata
    # ... other metadata
}
```

### Data Type Guidelines

**Inputs**: Can be tensors or dictionaries (e.g., point clouds with pos/feat structure)
**Labels**: Should be tensors 
**Meta_info**: Can be any JSON-serializable types (str, int, float, list, dict)

## Device Handling Philosophy

### Pylon's Device Transfer Strategy

**Rule**: Datasets create tensors on CPU, BaseDataset handles device transfer automatically.

```python
# ✅ CORRECT - Create on CPU
def _load_datapoint(self, idx):
    # Load data on CPU
    image = load_image(path)  # Returns CPU tensor
    features = torch.ones(num_points, 1)  # CPU tensor by default
    
    return inputs, labels, meta_info

# ❌ WRONG - Manual device handling  
def _load_datapoint(self, idx):
    image = load_image(path).to(self.device)  # Don't do this!
```

**Why**: BaseDataset handles device transfer intelligently to support multiprocessing DataLoaders.

## Dataset Size Validation

### DATASET_SIZE Constants

The `DATASET_SIZE` dictionary is **functionally critical**, not just documentation:

```python
DATASET_SIZE = {
    'train': 14313,  # Must match len(dataset) when split='train'
    'val': 915,      # Must match len(dataset) when split='val'  
    'test': 1520,    # Must match len(dataset) when split='test'
}
```

### Automatic Validation

BaseDataset automatically validates dataset size on initialization:

```python
def _init_annotations_all_splits(self):
    # ... load annotations ...
    actual_size = len(self)
    expected_size = self.DATASET_SIZE[self.split]
    assert actual_size == expected_size, \
        f"len(self)={actual_size}, self.DATASET_SIZE[self.split]={expected_size}"
```

**Critical**: If your filtering logic changes, you must update DATASET_SIZE constants.

### Generating Accurate Constants

Always generate constants programmatically:

```python
# Script to generate DATASET_SIZE constants
for split in ['train', 'val', 'test']:
    dataset = YourDataset(split=split, **params)
    print(f"'{split}': {len(dataset)},")
```

## Inheritance Patterns

### Single Dataset Class

For simple datasets with one variant:

```python
class SimpleDataset(BaseDataset):
    DATASET_SIZE = {...}
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
```

### Dataset Family Pattern

For datasets with multiple variants (e.g., 3DMatch/3DLoMatch):

```python
class _BaseDatasetFamily(BaseDataset):
    """Private base class - starts with underscore."""
    
    def __init__(self, filter_param=None, **kwargs):
        self.filter_param = filter_param
        super().__init__(**kwargs)
    
    def _init_annotations(self):
        # Common metadata loading logic
        # Apply self.filter_param for filtering
        pass
    
    def _load_datapoint(self, idx):
        # Common data loading logic
        pass

class SpecificDatasetA(_BaseDatasetFamily):
    """Public dataset class."""
    DATASET_SIZE = {...}  # Specific to this variant
    
    def __init__(self, **kwargs):
        super().__init__(filter_param=specific_value_a, **kwargs)

class SpecificDatasetB(_BaseDatasetFamily):
    """Public dataset class.""" 
    DATASET_SIZE = {...}  # Specific to this variant
    
    def __init__(self, **kwargs):
        super().__init__(filter_param=specific_value_b, **kwargs)
```

**When to use**: When multiple datasets share loading logic but have different filtering criteria.

## Error Handling Philosophy

### Fail Fast with Clear Messages

Pylon follows a "fail fast" philosophy - use assertions for validation:

```python
def _init_annotations(self):
    # File existence
    assert os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}"
    
    # Data structure validation
    assert isinstance(data, list), f"Expected list format, got {type(data)}"
    assert len(data) > 0, "Metadata list is empty"
    
    # Required keys
    expected_keys = ['key1', 'key2', 'key3']
    first_item = data[0]
    assert all(key in first_item for key in expected_keys), \
        f"Missing keys in metadata: {list(first_item.keys())}"
```

### When NOT to Use Try-Catch

**Don't use try-catch for defensive programming**:

```python
# ❌ WRONG - Hiding bugs
try:
    result = load_data(path)
except Exception:
    return None  # Masks the real problem

# ✅ CORRECT - Let it fail with clear message
result = load_data(path)  # Will crash if path invalid - GOOD!
```

**Use try-catch only for legitimate error recovery**:

```python
# ✅ CORRECT - Cache corruption recovery
try:
    with open(cache_file, 'rb') as f:
        cached_data = pickle.load(f)
except:
    # Cache corrupted, recompute
    cached_data = expensive_computation()
```

## Memory and Performance Patterns

### Lazy Loading Strategy

Datasets use two-phase loading for memory efficiency:

1. **Initialization**: Load only lightweight metadata into `self.annotations`
2. **On-demand**: Load actual data in `_load_datapoint()` when accessed

```python
def _init_annotations(self):
    # Only store file paths and metadata - NO actual data
    self.annotations = [
        {'image_path': '/path/img1.jpg', 'label': 5},
        {'image_path': '/path/img2.jpg', 'label': 3},
        # ... lightweight metadata only
    ]

def _load_datapoint(self, idx):
    annotation = self.annotations[idx]
    # Load actual data only when needed
    image = load_image(annotation['image_path'])
    return inputs, labels, meta_info
```

### Thread Safety Requirements

Datasets must be thread-safe for multi-worker DataLoaders:

- **Safe**: Reading from `self.annotations`, creating new tensors
- **Unsafe**: Modifying shared state, caching without locks

If you need caching, use proper synchronization or create cache keys that avoid collisions.

## Integration with Pylon Framework

### Transform Compatibility

Datasets must work with Pylon's transform system:

```python
# Transforms are applied after _load_datapoint()
inputs, labels, meta_info = dataset._load_datapoint(idx)
inputs, labels = transforms(inputs, labels)  # Applied by BaseDataset
```

**Key point**: Design your data format to be compatible with expected transforms.

### DataLoader Integration

Datasets work with PyTorch DataLoaders through BaseDataset:

```python
dataset = YourDataset(split='train')
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Requires thread safety
    collate_fn=your_collator
)
```

## Common Patterns and Examples

### Point Cloud Datasets

```python
inputs = {
    'src_pc': {
        'pos': torch.Tensor,  # (N, 3) positions
        'feat': torch.Tensor, # (N, F) features
    },
    'tgt_pc': {
        'pos': torch.Tensor,  # (M, 3) positions  
        'feat': torch.Tensor, # (M, F) features
    },
    'correspondences': torch.Tensor,  # (K, 2) point indices
}
```

### Image Classification

```python
inputs = {
    'image': torch.Tensor,  # (C, H, W) image
}

labels = {
    'class': torch.Tensor,  # (,) class index
}
```

### Change Detection

```python
inputs = {
    'image1': torch.Tensor,  # (C, H, W) before image
    'image2': torch.Tensor,  # (C, H, W) after image
}

labels = {
    'change_mask': torch.Tensor,  # (H, W) binary mask
}
```

## Debugging and Validation

### Dataset Inspection

Use these patterns for debugging:

```python
# Check dataset size
print(f"Dataset size: {len(dataset)}")

# Inspect a datapoint
inputs, labels, meta_info = dataset[0]
print(f"Inputs keys: {inputs.keys()}")
print(f"Labels keys: {labels.keys()}")
print(f"Meta info: {meta_info}")

# Validate tensor shapes and types
for key, tensor in inputs.items():
    print(f"{key}: {tensor.shape}, {tensor.dtype}")
```

### Common Issues

1. **DATASET_SIZE mismatch**: Update constants when filtering logic changes
2. **Device errors**: Don't manually transfer to device in _load_datapoint
3. **Memory issues**: Use lazy loading, don't store actual data in annotations
4. **Thread safety**: Avoid shared mutable state
5. **Transform incompatibility**: Check tensor shapes and dictionary structure
