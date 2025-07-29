# Dataset Implementation Guide for Pylon

## Overview

This guide teaches how to implement a new dataset in the Pylon framework, based on lessons learned from real-world dataset implementations. It covers the complete workflow from research to production-ready implementation.

## Table of Contents

1. [Pre-Implementation Research](#pre-implementation-research)
2. [Data Acquisition and Analysis](#data-acquisition-and-analysis)
3. [Implementation Strategy](#implementation-strategy)
4. [Dataset Family Patterns](#dataset-family-patterns)
5. [Cache Version Implementation](#cache-version-implementation)
6. [Performance Optimization](#performance-optimization)
7. [Documentation](#documentation)
8. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
9. [Lessons Learned](#lessons-learned)

## Related Documents

- **[Dataset Classes Design](dataset_classes_design.md)**: Understanding Pylon's BaseDataset architecture and patterns
- **[Dataset Testing Design](dataset_testing_design.md)**: Comprehensive testing strategies for dataset implementations
- **[Dataset Viewer Integration](dataset_viewer_integration.md)**: Guide for integrating datasets with Pylon's data viewer

---

## Pre-Implementation Research

### 1. Study Multiple Reference Implementations

**Why**: Different implementations reveal different design choices and edge cases.

**What to do**:
- Identify 2-3 official implementations of the target dataset
- Focus on well-maintained repositories with active communities
- Look for implementations in different frameworks (PyTorch, TensorFlow, etc.)

**Example from 3DMatch**:
```bash
# Primary references studied:
- OverlapPredator (official 3DMatch implementation)
- GeoTransformer (state-of-the-art registration)
- Official 3DMatch benchmark code

# Key differences discovered:
- Metadata format: OverlapPredator uses dict-of-arrays, GeoTransformer uses list-of-dicts
- Filtering strategy: Some apply overlap filtering at runtime, others pre-filter
- Test set curation: Different test files with different overlap distributions
```

**Critical Questions to Answer**:
- How do I download their data and/or trained model weights? Provide me a download instruction.
- In which files are the dataset classes defined? Give me a list.
- How do they handle train/val/test splits? What are the splits on this dataset? How are the splits defined? Give me a statistics summary in the dataset document at the end.
- After you download the data, traverse through the folder and see: which files contains the inputs, and which files contain the labels?
- What data format do they use for meta info? You need to read/load and inspect.
- What preprocessing steps did the authors apply to the data? In which file or from which reference are the preprocessing steps defined? Is the dataset class definition working with preprocessed data or raw data? Is the downloaded data the raw data from open source or the data preprocessed by the authors? Do I need to run the preprocessing myself? Can I trust the downloaded data?
- In the definition of the dataset class, when the datapoints are being loaded, what are some transforms or data augmentations applied to the raw data loaded from disk? List all of them. Which of them are already implemented in Pylon, which of them are additional transforms that Pylon needs to implement? Do a deep dive into the @data/transforms module and investigate. Think carefully and compare the Pylon data transforms implementation with the transforms applied in the official code.
- How do they compute derived data (e.g., correspondences for point cloud registration)? Is this expensive computation or no? If yes, think about creating a caching system, with strings as the cache keys. You should not be modifying anything in self.data_root. For all the cache files, you should be saving to a sibling folder of self.data_root.

---

## Data Acquisition and Analysis

### 1. Obtain Real Dataset Files

**Never assume dummy data is sufficient**. Always work with real metadata files.

**Process**:
1. **Download official data** from primary source repository
2. **Verify file integrity** (checksums if available)
3. **Compare metadata formats** across different sources
4. **Document discrepancies** between sources

### 2. Comprehensive Metadata Analysis

**Create analysis scripts** to understand the data structure:

```python
def analyze_metadata(metadata_file):
    """Analyze dataset metadata comprehensively."""
    with open(metadata_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"File: {metadata_file}")
    print(f"Type: {type(data)}")
    print(f"Total items: {len(data)}")
    
    if isinstance(data, list) and len(data) > 0:
        # List of dictionaries format
        print(f"Keys: {list(data[0].keys())}")
        # Extract key statistics based on your data structure
    elif isinstance(data, dict):
        # Dictionary format  
        print(f"Keys: {list(data.keys())}")
        # Extract key statistics based on your data structure
    
    # Add domain-specific analysis
    # e.g., for images: resolution distribution, class distribution
    # e.g., for point clouds: overlap distribution, scene analysis
```

### 3. Dataset Statistics Verification

**Critical**: Generate `DATASET_SIZE` constants from actual data analysis.

**Verification Process**:
```python
# Test your filtering logic matches expected sizes
dataset = YourDataset(split='train', **filter_params)
actual_size = len(dataset)
expected_size = dataset.DATASET_SIZE['train']

assert actual_size == expected_size, f"Size mismatch: {actual_size} != {expected_size}"
```

---

## Implementation Strategy

### 1. Framework Integration

See **[Dataset Classes Design](dataset_classes_design.md)** for complete details on:
- BaseDataset inheritance patterns
- Three-dictionary return structure (inputs, labels, meta_info)
- Device handling philosophy
- DATASET_SIZE validation requirements
- Data organization and soft links strategy

### 2. Metadata Format Handling

**Be prepared for format variations across implementations**:

```python
def _load_metadata(self, metadata_file):
    """Load metadata with format detection."""
    with open(metadata_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        # Format A: dict of arrays
        return self._process_dict_format(data)
    elif isinstance(data, list):
        # Format B: list of dicts
        return self._process_list_format(data)
    else:
        raise ValueError(f"Unknown metadata format: {type(data)}")

def _process_dict_format(self, data):
    """Process dictionary-of-arrays format."""
    # Convert to common internal format
    items = []
    for i in range(len(data['key1'])):
        item = {key: data[key][i] for key in data.keys()}
        items.append(item)
    return items

def _process_list_format(self, data):
    """Process list-of-dictionaries format.""" 
    # Already in preferred format
    return data
```

### 3. Robust Error Handling

**Pylon Philosophy**: Fail fast with clear error messages, don't hide bugs.

```python
def _init_annotations(self):
    # Assert file existence with clear message
    assert os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}"
    
    # Validate data structure
    assert isinstance(metadata_list, list), f"Expected list format, got {type(metadata_list)}"
    assert len(metadata_list) > 0, "Metadata list is empty"
    
    # Validate required keys with helpful error message
    expected_keys = ['key1', 'key2', 'key3']  # Adjust for your dataset
    first_item = metadata_list[0]
    assert all(key in first_item for key in expected_keys), \
        f"Missing keys in metadata: expected {expected_keys}, got {list(first_item.keys())}"
    
    # Domain-specific validation with context
    for i, item in enumerate(metadata_list):
        # Add validations specific to your dataset
        assert item['key1'] is not None, f"Invalid key1 in item {i}: {item}"
```

---

## Dataset Family Patterns

### When to Use Inheritance

Use the dataset family pattern when you have:
- Multiple variants of the same dataset (e.g., different filtering criteria)
- Shared data loading logic
- Different DATASET_SIZE constants per variant

### Implementation Example

```python
class _BaseDatasetFamily(BaseDataset):
    """Private base class for dataset family."""
    
    def __init__(self, filter_param=None, **kwargs):
        self.filter_param = filter_param
        super().__init__(**kwargs)
    
    def _init_annotations(self):
        # Common metadata loading logic
        raw_data = self._load_metadata()
        
        # Apply variant-specific filtering
        self.annotations = []
        for item in raw_data:
            if self._should_include_item(item):
                self.annotations.append(item)
    
    def _should_include_item(self, item):
        """Override in subclasses for specific filtering."""
        if self.filter_param is None:
            return True
        # Apply filter_param logic
        return item['some_value'] > self.filter_param
    
    def _load_datapoint(self, idx):
        # Common data loading logic
        annotation = self.annotations[idx]
        # ... load and process data
        return inputs, labels, meta_info

class HighQualityDataset(_BaseDatasetFamily):
    """Dataset with high-quality samples only."""
    DATASET_SIZE = {'train': 1000, 'val': 100, 'test': 200}
    
    def __init__(self, **kwargs):
        super().__init__(filter_param=0.8, **kwargs)

class AllDataset(_BaseDatasetFamily):
    """Dataset with all samples."""
    DATASET_SIZE = {'train': 5000, 'val': 500, 'test': 1000}
    
    def __init__(self, **kwargs):
        super().__init__(filter_param=None, **kwargs)
```

---

## Performance Optimization

### Expensive Computation Caching

When your dataset involves expensive computations during initialization or loading, implement intelligent caching:

**General Caching Pattern**:
```python
def _get_cached_expensive_computation(self, input_params):
    """Generic pattern for caching expensive computations."""
    # Create cache directory (sibling to data_root to avoid conflicts)
    cache_dir = os.path.join(
        os.path.dirname(self.data_root), 
        f'{os.path.basename(self.data_root)}_cache'
    )
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create simple, collision-resistant cache key
    cache_key = self._generate_cache_key(input_params)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Try cache first
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            return self._post_process_cached_result(result)
        except:
            pass  # Cache corrupted, recompute
    
    # Compute and cache
    result = self._expensive_computation(input_params)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(self._prepare_for_caching(result), f)
    except:
        pass  # Cache write failed, continue anyway
    
    return result

def _generate_cache_key(self, input_params):
    """Generate collision-resistant cache key."""
    # Example: combine relevant parameters
    param1, param2 = input_params
    return f"{param1}_{param2}_{self.some_config_param}"

def _expensive_computation(self, input_params):
    """The actual expensive computation."""
    # Example: correspondence computation, feature extraction, etc.
    pass

def _prepare_for_caching(self, result):
    """Prepare result for caching (e.g., move to CPU)."""
    if isinstance(result, torch.Tensor):
        return result.cpu().numpy()
    return result

def _post_process_cached_result(self, cached_result):
    """Convert cached result back to expected format."""
    if isinstance(cached_result, np.ndarray):
        return torch.tensor(cached_result, dtype=torch.int64, device=self.device)
    return cached_result
```

**Example: Point Cloud Correspondence Caching**:
```python
def _get_cached_correspondences(self, annotation, src_points, tgt_points, transform):
    """Cache expensive correspondence computation."""
    def compute_correspondences():
        return get_correspondences(src_points, tgt_points, transform, self.matching_radius)
    
    cache_key = f"{annotation['src_id']}_{annotation['tgt_id']}_{self.matching_radius}"
    return self._get_cached_expensive_computation((cache_key, compute_correspondences))
```

### Memory Efficiency

Follow lazy loading patterns:
```python
def _init_annotations(self):
    # Store only lightweight metadata - NO actual data
    self.annotations = [
        {'file_path': path, 'label': label, 'metadata': {...}}
        for path, label in metadata_items
    ]

def _load_datapoint(self, idx):
    annotation = self.annotations[idx]
    # Load actual data only when accessed
    raw_data = load_data(annotation['file_path'])
    # Return raw data - transforms are applied by framework
    return process_raw_data(raw_data)
```

---

## Documentation

### 1. Documentation Structure

Follow the established pattern from other Pylon datasets:

```markdown
# Dataset Name

## Overview
Brief description and purpose

## Quick Start  
### Basic Usage
Working code examples that users can copy-paste

### Custom Parameters
Advanced usage patterns

## Dataset Statistics
### Dataset Sizes Summary
Exact numbers with filtering explanations

## Data Format
### Input/Output Structure
Exact tensor shapes and types

## Implementation Details
### Key Features
Performance optimizations, caching, etc.

## References
Academic papers and implementations
```

### 2. Programmatic Statistics Generation

**Critical**: Documentation must match implementation exactly.

**Best Practice**:
```python
# Create script to generate documentation statistics
def generate_stats():
    for split in ['train', 'val', 'test']:
        dataset = YourDataset(split=split)
        print(f"| **{split}** | {len(dataset)} | Description |")

# Run this whenever DATASET_SIZE changes
```

### 3. Transparent Issue Documentation

**Be honest about dataset limitations**:

```markdown
## ⚠️ Known Issues

### Issue Name
**Problem**: Clear description of the issue
**Impact**: How it affects users/results  
**Mitigation**: What users can do about it
```

---

## Common Pitfalls and Solutions

### 1. DATASET_SIZE Mismatch

**Problem**: Constants don't match actual filtered sizes.

**Solution**: 
- Generate constants programmatically from actual data
- Add tests that verify constants match filtered sizes
- Update constants whenever filtering logic changes

### 2. Metadata Format Assumptions

**Problem**: Implementation assumes one format, data uses another.

**Solution**:
- Support multiple formats with detection logic
- Test with data from multiple sources
- Document which formats are supported

### 3. Device Handling Confusion

**Problem**: Manual device transfer breaks multiprocessing.

**Solution**: See [Dataset Classes Design](dataset_classes_design.md) for Pylon's device handling patterns.

### 4. Performance Issues

**Problem**: Dataset loading becomes a bottleneck.

**Solution**:
- Implement caching for expensive computations
- Use lazy loading patterns
- Profile memory usage and optimize hotspots

### 5. Test Data vs Real Data

**Problem**: Implementation works with test data but fails with real data.

**Solution**:
- Always use real dataset files for development
- Create analysis scripts to understand data structure
- Verify statistics match expectations

---

## Lessons Learned

### 1. Investigation is Critical

**Key Insight**: Spend 80% of time understanding the data, 20% implementing.

**Why**: Dataset implementations evolve across research groups and time. Without thorough investigation, you'll implement the wrong version or miss critical details.

**Process**:
1. Study 3+ reference implementations
2. Download and analyze real data files
3. Document all format variations discovered
4. Verify statistics with multiple sources

### 2. Fail Fast Philosophy

**Key Insight**: Assertions are better than defensive programming.

**Why**: Hidden bugs are worse than visible crashes. When data doesn't match expectations, crash immediately with clear error messages.

```python
# ✅ GOOD - Crash immediately with clear message
assert src_scene == tgt_scene, f"Scene mismatch: {src_scene} != {tgt_scene}"

# ❌ BAD - Hide the bug
if src_scene != tgt_scene:
    src_scene = tgt_scene  # Masks the real problem
```

### 3. Format Evolution Awareness

**Key Insight**: Dataset formats evolve; plan for multiple versions.

**Why**: The "same" dataset may have different metadata formats depending on the source.

**Solution**: Build format detection and conversion logic from the start.

### 4. Statistics Must Be Exact

**Key Insight**: DATASET_SIZE constants are functional requirements, not just documentation.

**Why**: Pylon validates dataset size against these constants. Wrong constants = broken tests.

**Process**: Always generate constants programmatically from actual data.

### 5. Real Data First

**Key Insight**: Never trust dummy/test data for implementation decisions.

**Why**: Test data often has simplified structure that doesn't reflect real-world complexity.

**Rule**: Get real metadata files before writing implementation code.

### 6. Performance from Day One

**Key Insight**: Design for performance early, don't treat it as an afterthought.

**Why**: Dataset loading is often the bottleneck in training pipelines.

**Critical Areas**:
- Expensive computation caching
- Memory-efficient data structures
- Thread-safe operations
- Lazy loading patterns

---

## Cache Version Implementation

### Overview

All Pylon datasets **must** implement cache versioning to ensure different dataset configurations use separate cache directories. This prevents cache collisions and data corruption.

### 1. Implementing `_get_cache_version_dict()`

**Every dataset class must override this method** to include parameters that affect dataset content:

```python
def _get_cache_version_dict(self) -> Dict[str, Any]:
    """Return parameters that affect dataset content for cache versioning."""
    # ALWAYS call parent implementation first
    version_dict = super()._get_cache_version_dict()
    
    # Add dataset-specific parameters that affect content
    version_dict.update({
        'param1': self.param1,
        'param2': self.param2,
        # Include ALL parameters that change dataset content
    })
    
    return version_dict
```

### 2. Hierarchical Version Implementation

**Follow the inheritance pattern**:

```python
# BaseDataset provides: class_name, data_root, split/split_percentages
class BaseDataset:
    def _get_cache_version_dict(self) -> Dict[str, Any]:
        return {
            'class_name': self.__class__.__name__,
            'data_root': str(self.data_root),
            'split': self.split,  # or split_percentages
        }

# SyntheticDataset adds: source info, dataset_size
class SyntheticDataset(BaseDataset):
    def _get_cache_version_dict(self) -> Dict[str, Any]:
        version_dict = super()._get_cache_version_dict()
        version_dict.update({
            'rotation_mag': self.rotation_mag,
            'translation_mag': self.translation_mag,
            'dataset_size': self.dataset_size,
        })
        return version_dict

# SpecificDataset adds: specific parameters
class LiDARCameraPoseDataset(SyntheticDataset):
    def _get_cache_version_dict(self) -> Dict[str, Any]:
        version_dict = super()._get_cache_version_dict()
        if self.camera_count is not None:
            version_dict['camera_count'] = self.camera_count
        return version_dict
```

### 3. Critical Implementation Rules

#### **Include ALL Content-Affecting Parameters**
```python
# ✅ CORRECT - Include everything that changes dataset content
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    version_dict.update({
        'dataset_size': self.dataset_size,           # Affects number of items
        'overlap_threshold': self.overlap_threshold, # Affects filtering
        'augmentation_mode': self.aug_mode,          # Affects data generation
        'file_paths': sorted(self.file_paths),       # Affects source data
    })
    return version_dict

# ❌ WRONG - Missing parameters that affect content
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    # Missing dataset_size, overlap_threshold, etc.
    return version_dict
```

#### **Handle Optional Parameters Correctly**
```python
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    
    # ✅ CORRECT - Only add if not None
    if self.camera_count is not None:
        version_dict['camera_count'] = self.camera_count
    
    # ✅ CORRECT - Always add required parameters
    version_dict['dataset_size'] = self.dataset_size
    
    return version_dict
```

#### **Ensure Deterministic Ordering**
```python
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    version_dict.update({
        # ✅ CORRECT - Sort file paths for deterministic hashing
        'file_paths': sorted(self.file_paths),
        
        # ✅ CORRECT - Convert complex objects to strings
        'transforms_config': str(self.transforms_config),
    })
    return version_dict
```

### 4. Testing Requirements

**Every dataset implementation must include discrimination tests**:

```python
def test_dataset_version_discrimination():
    """Test that dataset instances with different parameters have different version hashes."""
    
    # Same parameters should have same hash
    dataset1a = MyDataset(param1=value1, param2=value2)
    dataset1b = MyDataset(param1=value1, param2=value2)
    assert dataset1a.get_cache_version_hash() == dataset1b.get_cache_version_hash()
    
    # Different param1 should have different hash
    dataset2 = MyDataset(param1=different_value, param2=value2)
    assert dataset1a.get_cache_version_hash() != dataset2.get_cache_version_hash()
    
    # Different param2 should have different hash
    dataset3 = MyDataset(param1=value1, param2=different_value)
    assert dataset1a.get_cache_version_hash() != dataset3.get_cache_version_hash()
    
    # Test ALL parameters that should affect caching
```

### 5. Cache Directory Structure

The cache system automatically creates directory structure:
```
<data_root>_cache/
├── a7f8c9d2b4e6f1a3/    # Version hash for config A
│   ├── 0.pt
│   ├── 1.pt
│   └── ...
├── b2e4c8f1a6d9e5c7/    # Version hash for config B  
│   ├── 0.pt
│   ├── 1.pt
│   └── ...
└── cache_metadata.json  # Human-readable mapping
```

### 6. Common Implementation Mistakes

#### **❌ Missing Content-Affecting Parameters**
```python
# WRONG - Missing dataset_size affects number of generated items
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    # Missing self.dataset_size - CACHE COLLISION RISK!
    return version_dict
```

#### **❌ Including Non-Content Parameters**
```python
# WRONG - Including parameters that don't affect dataset content
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    version_dict.update({
        'num_workers': self.num_workers,    # WRONG - doesn't affect data
        'batch_size': self.batch_size,      # WRONG - doesn't affect data
        'dataset_size': self.dataset_size,  # ✅ CORRECT - affects data
    })
    return version_dict
```

#### **❌ Not Calling Parent Implementation**
```python
# WRONG - Must call super() to get base parameters
def _get_cache_version_dict(self) -> Dict[str, Any]:
    return {
        'my_param': self.my_param,
        # MISSING: class_name, data_root, split from parent
    }

# ✅ CORRECT
def _get_cache_version_dict(self) -> Dict[str, Any]:
    version_dict = super()._get_cache_version_dict()
    version_dict['my_param'] = self.my_param
    return version_dict
```

---

## Data Viewer Integration

### Registering Datasets with the Viewer

Once your dataset is implemented, register it with Pylon's data viewer for visual inspection and debugging:

#### 1. Add to Dataset Groups

Update `/data/viewer/backend/backend.py` to include your dataset in the appropriate group:

```python
DATASET_GROUPS = {
    'semseg': ['coco_stuff_164k'],
    '2dcd': ['air_change', 'cdd', 'levir_cd', 'oscd', 'sysu_cd'],
    '3dcd': ['urb3dcd', 'slpccd'],
    'pcr': ['synth_pcr', 'real_pcr', 'kitti', 'your_new_dataset'],  # Add here
}
```

**Dataset Type Guidelines**:
- `semseg`: Semantic segmentation datasets (single image → label map)
- `2dcd`: 2D change detection datasets (two images → change map)  
- `3dcd`: 3D change detection datasets (two point clouds → change map)
- `pcr`: Point cloud registration datasets (two point clouds → transformation)

#### 2. Create Viewer Configuration

Create a config file in the appropriate directory:

**Path Pattern**: `/configs/common/datasets/{domain}/train/{dataset_name}_data_cfg.py`

**Example for PCR dataset**:
```python
# /configs/common/datasets/point_cloud_registration/train/your_dataset_data_cfg.py
import torch
import data
import utils

data_cfg = {
    'train_dataset': {
        'class': data.datasets.YourDataset,
        'args': {
            'data_root': './data/datasets/soft_links/your_data',
            'split': 'train',
            # Add dataset-specific parameters
        },
    },
    'train_dataloader': {
        'class': torch.utils.data.DataLoader,
        'args': {
            'batch_size': 1,  # Use batch_size=1 for viewer
            'num_workers': 4,
            'shuffle': True,
        },
    },
    'criterion': None,  # Not needed for viewer
}
```

**Important Notes**:
- **Batch size must be 1** for the viewer to work correctly
- **File name must match pattern**: `{dataset_name}_data_cfg.py`
- **Config structure must match** the training config format

#### 3. Viewer Adaptation to Dataset API

The data viewer automatically adapts to your dataset's existing API - you don't need to modify your dataset implementation. The viewer detects the dataset type and handles the data format appropriately.

**Key Principle**: The viewer adapts to your dataset, not the other way around.

The viewer will automatically handle whatever format your dataset returns for the `inputs`, `labels`, and `meta_info` dictionaries according to Pylon's three-dictionary pattern.

#### 4. Test Viewer Integration

Launch the viewer and verify your dataset appears:

```bash
# Launch data viewer
python -m data.viewer.cli

# Your dataset should appear in the dropdown as:
# [PCR] your_dataset_name
```

**Troubleshooting**:
- **Dataset not appearing**: Check config file name and location
- **Loading errors**: Verify data paths and dataset parameters
- **Display issues**: Ensure data format matches viewer expectations
- **Import errors**: Check that all required modules are imported in config

For detailed viewer-specific considerations, performance optimization, and troubleshooting, see the **[Dataset Viewer Integration Guide](dataset_viewer_integration.md)**.

---

## Conclusion

Implementing a dataset in Pylon requires:

1. **Thorough Research**: Understanding format variations and implementations
2. **Real Data Analysis**: Working with actual files, not dummy data  
3. **Framework Integration**: Following Pylon's patterns (see [Dataset Classes Design](dataset_classes_design.md))
4. **Robust Testing**: Comprehensive validation (see [Dataset Testing Design](dataset_testing_design.md))
5. **Performance Focus**: Caching and optimization from the start
6. **Accurate Documentation**: Statistics that match implementation
7. **Viewer Integration**: Register with data viewer for visual debugging (see [Dataset Viewer Integration Guide](dataset_viewer_integration.md))

The key insight is that dataset implementation is **data archaeology** - understanding how a dataset has evolved across research groups and implementations over time.

Success comes from spending most of your time understanding the data landscape before writing implementation code.
