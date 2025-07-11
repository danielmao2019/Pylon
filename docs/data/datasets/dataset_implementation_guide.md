# Dataset Implementation Guide for Pylon

## Overview

This guide teaches how to implement a new dataset in the Pylon framework, based on lessons learned from implementing the 3DMatch dataset family. It covers the complete workflow from research to production-ready implementation.

## Table of Contents

1. [Pre-Implementation Research](#pre-implementation-research)
2. [Understanding Pylon Framework](#understanding-pylon-framework)
3. [Data Acquisition and Analysis](#data-acquisition-and-analysis)
4. [Implementation Strategy](#implementation-strategy)
5. [Testing and Validation](#testing-and-validation)
6. [Documentation](#documentation)
7. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
8. [Lessons Learned](#lessons-learned)

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
- What data format do they use for metadata?
- How do they handle train/val/test splits?
- What preprocessing steps do they apply?
- How do they compute derived data (e.g., correspondences)?
- What are the exact dataset statistics (number of pairs, overlap distributions)?

### 2. Understand Dataset Variants and Evolution

**Key Insight**: Dataset implementations evolve over time, creating multiple "versions" of the same dataset.

**Investigation Steps**:
1. **Timeline Analysis**: When was each implementation created?
2. **Version Tracking**: Are there different versions of metadata files?
3. **Community Usage**: Which version is most commonly used in recent papers?

**3DMatch Example**:
- Original 3DMatch (2017): Basic overlap-based filtering
- 3DLoMatch variant (2021): Focus on low-overlap scenarios  
- GeoTransformer version (2022): Updated metadata format and filtering

---

## Understanding Pylon Framework

### 1. BaseDataset Architecture

**Critical Concepts to Master**:

```python
class YourDataset(BaseDataset):
    # Required class attributes
    SPLIT_OPTIONS = ['train', 'val', 'test']  # Supported splits
    INPUT_NAMES = ['src_pc', 'tgt_pc']        # Input tensor names
    LABEL_NAMES = ['transform']               # Label tensor names  
    SHA1SUM = None                           # Optional data integrity check
    DATASET_SIZE = {                         # CRITICAL: Must match actual filtered sizes
        'train': 12345,
        'val': 678, 
        'test': 910
    }

    def __init__(self, **kwargs):
        # Custom parameters first
        self.custom_param = kwargs.pop('custom_param', default_value)
        # Always call super().__init__(**kwargs) last
        super().__init__(**kwargs)

    def _init_annotations(self):
        # Load and filter metadata
        # Create self.annotations list
        pass

    def _load_datapoint(self, idx):
        # Return (inputs, labels, meta_info) tuple
        pass
```

### 2. Three-Dictionary Return Pattern

**MANDATORY Structure**:
```python
def _load_datapoint(self, idx):
    return inputs, labels, meta_info

# where:
inputs = {
    'src_pc': {'pos': tensor, 'feat': tensor},
    'tgt_pc': {'pos': tensor, 'feat': tensor},
    'correspondences': tensor,
}

labels = {
    'transform': tensor,  # 4x4 transformation matrix
}

meta_info = {
    'idx': idx,              # Added automatically by BaseDataset
    'src_path': str,         # Source file path
    'tgt_path': str,         # Target file path  
    'scene_name': str,       # Scene identifier
    'overlap': float,        # Overlap ratio
    'src_frame': int,        # Source fragment ID
    'tgt_frame': int,        # Target fragment ID
}
```

### 3. Device Handling Philosophy

**Pylon Rule**: Datasets create tensors on CPU, BaseDataset handles device transfer.

```python
# ✅ CORRECT - Create on CPU, let BaseDataset handle device transfer
src_pc_tensor = load_point_cloud(path)  # Returns CPU tensor
features = torch.ones((num_points, 1), dtype=torch.float32)  # CPU tensor

# ❌ WRONG - Manual device handling
src_pc_tensor = load_point_cloud(path).to(self.device)  # Don't do this
```

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
    print(f"Total pairs: {len(data)}")
    
    if isinstance(data, list):
        # GeoTransformer format
        print(f"Keys: {list(data[0].keys())}")
        overlaps = [item['overlap'] for item in data]
    else:
        # OverlapPredator format  
        print(f"Keys: {list(data.keys())}")
        overlaps = data['overlap']
    
    # Overlap distribution analysis
    print(f"Overlap range: {min(overlaps):.4f} - {max(overlaps):.4f}")
    print(f"Mean overlap: {np.mean(overlaps):.4f}")
    
    # Filter analysis for different thresholds
    for threshold in [0.1, 0.3, 0.5]:
        count = sum(1 for o in overlaps if o > threshold)
        print(f"Pairs with overlap > {threshold}: {count}")
```

### 3. Dataset Statistics Verification

**Critical**: The `DATASET_SIZE` constants must match actual filtered dataset sizes.

**Verification Process**:
```python
# Test your filtering logic
dataset = YourDataset(split='train', overlap_min=0.3, overlap_max=1.0)
actual_size = len(dataset)
expected_size = dataset.DATASET_SIZE['train']

assert actual_size == expected_size, f"Size mismatch: {actual_size} != {expected_size}"
```

---

## Implementation Strategy

### 1. Start with Base Class Design

**Design Pattern**: Use inheritance for dataset families.

```python
class _BaseDatasetFamily(BaseDataset):
    """Private base class for dataset family."""
    
    def __init__(self, custom_param=default, **kwargs):
        self.custom_param = custom_param
        super().__init__(**kwargs)
    
    def _init_annotations(self):
        # Common metadata loading logic
        pass
    
    def _load_datapoint(self, idx):
        # Common data loading logic
        pass

class SpecificDataset(_BaseDatasetFamily):
    """Public dataset with specific filtering."""
    
    DATASET_SIZE = {...}  # Specific to this variant
    
    def __init__(self, **kwargs):
        super().__init__(
            custom_param=specific_value,
            **kwargs
        )
```

### 2. Metadata Format Handling

**Be prepared for format variations**:

```python
def _load_metadata(self, metadata_file):
    """Load metadata with format detection."""
    with open(metadata_file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, dict):
        # OverlapPredator format: dict of arrays
        return self._process_dict_format(data)
    elif isinstance(data, list):
        # GeoTransformer format: list of dicts
        return self._process_list_format(data)
    else:
        raise ValueError(f"Unknown metadata format: {type(data)}")
```

### 3. Efficient Correspondence Caching

**Performance Critical**: Correspondence computation is expensive.

```python
def _get_cached_correspondences(self, annotation, src_points, tgt_points, transform):
    """Get correspondences with intelligent caching."""
    # Create cache directory (sibling to data_root)
    cache_dir = os.path.join(
        os.path.dirname(self.data_root), 
        f'{os.path.basename(self.data_root)}_correspondences_cache'
    )
    os.makedirs(cache_dir, exist_ok=True)
    
    # Simple, collision-resistant cache key
    src_name = os.path.basename(annotation['src_path']).split('.')[0]
    tgt_name = os.path.basename(annotation['tgt_path']).split('.')[0]
    cache_key = f"{src_name}_{tgt_name}_{self.matching_radius}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    # Try cache first, compute if miss
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                correspondences = pickle.load(f)
            return torch.tensor(correspondences, dtype=torch.int64, device=self.device)
        except:
            pass  # Cache corrupted, recompute
    
    # Compute and cache
    correspondences = get_correspondences(src_points, tgt_points, transform, self.matching_radius)
    
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(correspondences.cpu().numpy(), f)
    except:
        pass  # Cache write failed, continue anyway
    
    return correspondences
```

### 4. Robust Error Handling

**Pylon Philosophy**: Fail fast with clear error messages, don't hide bugs.

```python
def _init_annotations(self):
    # Assert file existence
    assert os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}"
    
    # Validate data structure
    assert isinstance(metadata_list, list), f"Expected list format, got {type(metadata_list)}"
    assert len(metadata_list) > 0, "Metadata list is empty"
    
    # Validate required keys
    expected_keys = ['pcd0', 'pcd1', 'rotation', 'translation', 'overlap']
    first_item = metadata_list[0]
    assert all(key in first_item for key in expected_keys), \
        f"Missing keys in metadata: {list(first_item.keys())}"
    
    # Validate scene name consistency
    for item in filtered_items:
        src_scene = item['pcd0'].split('/')[0]
        tgt_scene = item['pcd1'].split('/')[0]
        assert src_scene == tgt_scene, \
            f"Scene names must match: src={src_scene}, tgt={tgt_scene}"
```

---

## Testing and Validation

### 1. Comprehensive Test Structure

**Test File Organization**:
```
tests/data/datasets/your_domain/
├── test_your_dataset.py           # Main functionality tests
├── test_your_dataset_determinism.py  # Reproducibility tests
└── conftest.py                    # Shared test fixtures
```

### 2. Validation Functions

**Create thorough validation for each data type**:

```python
def validate_inputs(inputs: Dict[str, Any]) -> None:
    """Validate input dictionary structure and types."""
    assert isinstance(inputs, dict)
    assert inputs.keys() == {'src_pc', 'tgt_pc', 'correspondences'}
    
    # Validate point clouds
    for pc_name in ['src_pc', 'tgt_pc']:
        pc = inputs[pc_name]
        assert isinstance(pc, dict)
        assert 'pos' in pc and 'feat' in pc
        
        # Position validation
        assert isinstance(pc['pos'], torch.Tensor)
        assert pc['pos'].ndim == 2 and pc['pos'].shape[1] == 3
        assert pc['pos'].dtype == torch.float32
        
        # Feature validation  
        assert isinstance(pc['feat'], torch.Tensor)
        assert pc['feat'].ndim == 2 and pc['feat'].shape[1] == 1
        assert pc['feat'].dtype == torch.float32
        assert pc['pos'].shape[0] == pc['feat'].shape[0]
    
    # Validate correspondences
    correspondences = inputs['correspondences']
    assert isinstance(correspondences, torch.Tensor)
    assert correspondences.ndim == 2 and correspondences.shape[1] == 2
    assert correspondences.dtype == torch.int64
```

### 3. Dataset Size Validation

**Critical Test**: Verify DATASET_SIZE constants are correct.

```python
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_dataset_size_constants(split):
    """Test that DATASET_SIZE constants match actual filtered sizes."""
    dataset = YourDataset(split=split, data_root=TEST_DATA_ROOT)
    
    actual_size = len(dataset)
    expected_size = dataset.DATASET_SIZE[split]
    
    assert actual_size == expected_size, \
        f"Dataset size mismatch for {split}: {actual_size} != {expected_size}"
```

### 4. Parallel Testing for Performance

**Use threading for testing multiple samples**:

```python
def test_dataset_samples(dataset, max_samples=5):
    """Test dataset samples in parallel."""
    
    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)
    
    # Test random samples
    num_samples = min(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
```

---

## Documentation

### 1. Documentation Structure

**Follow established pattern**:

```markdown
# Dataset Name

## Overview
Brief description and purpose

## Quick Start  
### Basic Usage
Working code examples

### Custom Parameters
Advanced usage patterns

## Dataset Statistics
### Dataset Sizes Summary
Exact numbers with filtering explanations

### Metadata Files Overview
Description of each file

## Dataset Design
### Filtering Strategy
How train/val/test splits work

### Known Issues
Data leakage, overlaps, etc.

## Data Format
### Input/Output Structure
Exact tensor shapes and types

### File System Structure
Directory organization

## Implementation Details
### Metadata Format
Exact file structure with examples

### Key Features
Caching, thread safety, etc.

## References
Academic papers and implementations
```

### 2. Accurate Statistics

**Critical**: Documentation must match implementation.

**Process**:
1. **Generate statistics programmatically** from actual data
2. **Verify numbers match DATASET_SIZE constants**
3. **Update documentation when constants change**
4. **Include example outputs** that users can reproduce

### 3. Known Issues Documentation

**Be transparent about dataset limitations**:

```markdown
## ⚠️ Known Issues

### Data Leakage Between Train/Val
**Problem**: Scene X appears in both splits
**Impact**: May inflate validation metrics  
**Mitigation**: Use test set for final evaluation

### Cross-Dataset Overlap  
**Problem**: 103 pairs appear in both test sets
**Impact**: 6.8% overlap in evaluation
**Mitigation**: Minimal impact on evaluation validity
```

---

## Common Pitfalls and Solutions

### 1. DATASET_SIZE Mismatch

**Problem**: Constants don't match actual filtered sizes.

**Root Cause**: Using statistics from different dataset version.

**Solution**: 
- Always generate constants from the actual data you're using
- Create verification tests that fail when constants are wrong
- Document which dataset version/source your constants come from

### 2. Metadata Format Assumptions

**Problem**: Implementation assumes one format, data uses another.

**Root Cause**: Not checking multiple reference implementations.

**Solution**:
- Support multiple formats in your implementation
- Add format detection logic
- Document which formats are supported

### 3. Device Handling Confusion

**Problem**: Manual device transfer breaks multiprocessing.

**Root Cause**: Not understanding Pylon's device handling philosophy.

**Solution**:
- Always create tensors on CPU in `_load_datapoint()`
- Let BaseDataset handle device transfer
- Never use `.to(device)` in dataset classes

### 4. Correspondence Computation Performance

**Problem**: Dataset loading is too slow.

**Root Cause**: Computing correspondences on every access.

**Solution**:
- Implement intelligent caching strategy
- Use simple, collision-resistant cache keys
- Handle cache corruption gracefully

### 5. Test Data vs Real Data Confusion

**Problem**: Implementation works with test data but fails with real data.

**Root Cause**: Using dummy/test metadata files instead of real dataset.

**Solution**:
- Always obtain real metadata files from official sources
- Verify file contents match expected structure
- Create analysis scripts to understand data format

---

## Lessons Learned

### 1. Investigation is Critical

**Key Insight**: Spend 60% of time understanding the data, 40% implementing.

**Why**: Dataset implementations in the wild have evolved over time, creating multiple "versions" of the same dataset. Without thorough investigation, you'll implement the wrong version or miss critical details.

**Process**:
1. Study 3+ reference implementations
2. Download and analyze real data files
3. Document all format variations discovered
4. Verify statistics with multiple sources

### 2. Fail Fast Philosophy

**Key Insight**: Assertions are better than defensive programming.

**Why**: Hidden bugs are worse than visible crashes. When data doesn't match expectations, crash immediately with clear error messages rather than trying to handle "impossible" cases.

**Examples**:
```python
# ✅ GOOD - Crash immediately with clear message
assert src_scene == tgt_scene, f"Scene mismatch: {src_scene} != {tgt_scene}"

# ❌ BAD - Hide the bug
if src_scene != tgt_scene:
    src_scene = tgt_scene  # Masks the real problem
```

### 3. Format Evolution Awareness

**Key Insight**: Dataset formats evolve; plan for multiple versions.

**Why**: The "same" dataset may have different metadata formats depending on which research group's implementation you use.

**Solution**: Build format detection and conversion logic from the start.

### 4. Statistics Must Be Exact

**Key Insight**: DATASET_SIZE constants are not just documentation - they're functional requirements.

**Why**: Pylon's BaseDataset validates actual dataset size against these constants. Wrong constants = broken tests.

**Process**: Always generate constants programmatically from the actual data.

### 5. Real Data First

**Key Insight**: Never trust dummy/test data for implementation decisions.

**Why**: Test data often has simplified structure that doesn't reflect real-world edge cases.

**Rule**: Get real metadata files before writing any implementation code.

### 6. Documentation as Code

**Key Insight**: Documentation should be generated from the same data as the implementation.

**Why**: Manual documentation gets out of sync with code changes.

**Best Practice**: Write scripts that analyze the data and generate documentation statistics.

### 7. Testing Strategy

**Key Insight**: Test the framework integration, not just the data loading.

**Why**: Dataset classes must work within Pylon's broader ecosystem (transforms, dataloaders, etc.).

**Focus Areas**:
- Tensor types and shapes
- Device handling  
- Multiprocessing compatibility
- Transform compatibility
- Memory usage patterns

### 8. Performance from Day One

**Key Insight**: Don't treat performance as an afterthought.

**Why**: Dataset loading is often the bottleneck in training pipelines.

**Critical Optimizations**:
- Correspondence caching
- Efficient metadata format
- Minimal memory allocation
- Thread-safe operations

---

## Conclusion

Implementing a dataset in Pylon requires:

1. **Thorough Research**: Understanding all format variations and implementations
2. **Real Data Analysis**: Working with actual dataset files, not dummy data  
3. **Framework Understanding**: Following Pylon's patterns and philosophies
4. **Robust Implementation**: Failing fast with clear error messages
5. **Comprehensive Testing**: Validating integration with the broader framework
6. **Accurate Documentation**: Statistics that match the actual implementation

The key insight is that dataset implementation is **data archaeology** - you're not just loading files, you're understanding how a dataset has evolved across multiple research groups and implementations over time.

Success comes from spending the majority of your time understanding the data landscape before writing any implementation code.
