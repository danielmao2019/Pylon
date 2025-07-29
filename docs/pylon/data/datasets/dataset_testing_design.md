# Dataset Testing Design Guide

## Overview

This document outlines the actual testing patterns used in Pylon for dataset implementations. These patterns are derived from analysis of existing test files and focus on practical validation approaches.

## Actual Testing Philosophy in Pylon

### Core Patterns Observed
1. **Assert-Based Validation**: Direct assertions for all validation, no pytest.skip patterns
2. **ThreadPoolExecutor Parallelism**: Standard pattern for testing multiple datapoints
3. **CLI-Controlled Sampling**: `--samples` flag controls test scope via `get_samples_to_test()`
4. **Fixture-Based Dataset Creation**: Heavy use of parametrized fixtures
5. **Mathematical Validation**: Complex checks for domain-specific constraints

### Testing Scope
- **Data Loading**: Verify tensor shapes, dtypes, and value ranges
- **API Contracts**: Ensure three-dictionary return format (inputs, labels, meta_info)
- **Mathematical Properties**: Domain-specific validation (transformations, class distributions)
- **Index Consistency**: Verify meta_info['idx'] matches datapoint index
- **Caching**: Comprehensive cache behavior testing when applicable

## Core Testing Patterns

### 1. Fixture-Based Dataset Creation

**Standard Pattern**: Use parametrized fixtures for dataset instantiation

```python
@pytest.fixture
def dataset(request):
    """Fixture for creating dataset instances."""
    params = request.param
    return MyDataset(**params)

@pytest.mark.parametrize('dataset', [
    {'split': 'train', 'data_root': './data/path'},
    {'split': 'val', 'data_root': './data/path'},
], indirect=True)
def test_my_dataset(dataset, max_samples, get_samples_to_test):
    # Test implementation...
```

### 2. ThreadPoolExecutor Validation

**Standard Pattern**: Parallel validation of multiple datapoints

```python
def test_dataset_validation(dataset, max_samples, get_samples_to_test):
    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict)
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)
    
    num_samples = get_samples_to_test(len(dataset), max_samples, default=3)
    indices = list(range(num_samples))  # or random.sample()
    with ThreadPoolExecutor() as executor:
        executor.map(validate_datapoint, indices)
```

### 3. CLI-Controlled Sampling

**Standard Pattern**: Use command-line `--samples` flag for test scope control

```python
def test_dataset_sampling(dataset, max_samples, get_samples_to_test):
    # This automatically respects --samples CLI argument
    num_samples = get_samples_to_test(len(dataset), max_samples, default=3)
    
    # For small datasets, test all samples
    if len(dataset) <= 10:
        indices = list(range(len(dataset)))
    else:
        # For large datasets, use random sampling
        indices = random.sample(range(len(dataset)), num_samples)
    
    # Validate all selected samples...
```

### 4. Assert-Based Validation

**Standard Pattern**: Direct assertions for all validation, no skip patterns

```python
def test_dataset_basic_validation():
    dataset = MyDataset(split='train')
    
    # Assert-based validation - no pytest.skip
    assert len(dataset) > 0, "Dataset should not be empty"
    
    # Test normal functionality
    datapoint = dataset[0]
    assert isinstance(datapoint, dict)
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
```

### 5. Mathematical Validation

**Standard Pattern**: Domain-specific mathematical checks for complex datasets

```python
def validate_transformation_matrix(transform: torch.Tensor, rot_mag: float, trans_mag: float):
    """Example from PCR dataset testing."""
    R = transform[:3, :3]
    t = transform[:3, 3]
    
    # Check rotation matrix properties
    assert torch.allclose(R @ R.T, torch.eye(3, device=R.device), atol=1e-6), \
        "Invalid rotation matrix: not orthogonal"
    assert torch.abs(torch.det(R) - 1.0) < 1e-6, \
        "Invalid rotation matrix: determinant not 1"
    
    # Check magnitude constraints
    rot_angle = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1, 1))
    assert torch.abs(rot_angle) <= np.radians(rot_mag)
    assert torch.norm(t) <= trans_mag

def validate_class_distribution(class_dist: torch.Tensor, dataset, num_samples: int):
    """Example from segmentation dataset testing."""
    if num_samples == len(dataset):
        if dataset.split == 'train':
            # Allow tolerance for training data
            assert abs(class_dist[1] / class_dist[0] - dataset.CLASS_DIST[1] / dataset.CLASS_DIST[0]) < 1.0e-02
        else:
            # Exact match for test data
            assert class_dist.tolist() == dataset.CLASS_DIST
```

### 6. Complex Parametrized Testing

**Standard Pattern**: Extensive parametrization for different configurations

```python
@pytest.mark.parametrize('dataset_with_params', [
    {
        'data_root': './data/path',
        'split': 'train',
        'voxel_size': 10.0,
        'transforms_cfg': transforms_cfg(rot_mag=45.0, trans_mag=0.5),
        'rot_mag': 45.0,  # For validation
        'trans_mag': 0.5,  # For validation
    },
    {
        'data_root': './data/path', 
        'split': 'val',
        'voxel_size': 5.0,
        'transforms_cfg': transforms_cfg(rot_mag=30.0, trans_mag=0.3),
        'rot_mag': 30.0,
        'trans_mag': 0.3,
    },
], indirect=True)
def test_complex_dataset(dataset_with_params, max_samples, get_samples_to_test):
    dataset, rot_mag, trans_mag = dataset_with_params
    # Test with extracted validation parameters...
```

## Standard Validation Functions

### Input Validation Pattern

```python
def validate_inputs(inputs: Dict[str, Any]) -> None:
    assert isinstance(inputs, dict), f"{type(inputs)=}"
    assert set(inputs.keys()) == expected_keys, f"{inputs.keys()=}"
    
    # Tensor-specific validation
    for key, tensor in inputs.items():
        assert isinstance(tensor, torch.Tensor), f"{key} should be torch.Tensor"
        assert tensor.dtype == expected_dtype, f"{key} dtype incorrect"
        assert not torch.isnan(tensor).any(), f"{key} contains NaN values"

def validate_labels(labels: Dict[str, Any]) -> None:
    assert isinstance(labels, dict), f"{type(labels)=}"
    assert set(labels.keys()) == expected_keys, f"{labels.keys()=}"
    # Add domain-specific label validation...

def validate_meta_info(meta_info: Dict[str, Any], datapoint_idx: int) -> None:
    assert isinstance(meta_info, dict), f"{type(meta_info)=}"
    assert 'idx' in meta_info, f"meta_info missing 'idx' key"
    assert meta_info['idx'] == datapoint_idx, f"Index mismatch: {meta_info['idx']=}, {datapoint_idx=}"
```

## Test Organization

### File Structure (Actual Pattern)
```
tests/data/datasets/
├── conftest.py                    # Critical - provides max_samples, get_samples_to_test fixtures
├── test_dataset_caching.py        # Comprehensive cache testing patterns
├── random_datasets/
│   └── test_classification_random_dataset.py
├── change_detection_datasets/
│   └── bi_temporal/
│       └── test_air_change_dataset.py
└── pcr_datasets/
    └── test_synth_pcr_dataset.py
```

### conftest.py Critical Fixtures

The `conftest.py` file provides essential test infrastructure:

```python
# conftest.py (key fixtures observed)
@pytest.fixture
def max_samples(request):
    """Extract --samples CLI argument for test control."""
    return getattr(request.config.option, 'samples', None)

@pytest.fixture  
def get_samples_to_test():
    """Function to determine number of samples to test based on CLI args."""
    def _get_samples(dataset_size: int, max_samples: int, default: int = None):
        if max_samples is not None:
            return min(max_samples, dataset_size)
        elif default is not None:
            return min(default, dataset_size)
        else:
            return dataset_size
    return _get_samples
```

## Caching Test Patterns (When Applicable)

For datasets with caching support, comprehensive cache testing follows specific patterns:

### Cache Functionality Testing

```python
def test_basic_cache_functionality(SampleDataset):
    """Test basic cache functionality - items are cached and retrievable."""
    dataset = SampleDataset(split='train', indices=list(range(10)), use_cache=True)
    
    # First access should cache items
    first_access = [dataset[i] for i in range(10)]
    # Second access should retrieve from cache  
    second_access = [dataset[i] for i in range(10)]
    
    # Verify items are identical
    for first, second in zip(first_access, second_access):
        assert buffer_equal(first, second)

def test_cache_vs_uncached(SampleDataset):
    """Test that cached and uncached datasets return identical data."""
    cached = SampleDataset(split='train', indices=list(range(5)), use_cache=True)
    uncached = SampleDataset(split='train', indices=list(range(5)), use_cache=False)
    
    for i in range(5):
        assert buffer_equal(cached[i], uncached[i])
```

## No-Patterns (Things NOT Used in Pylon)

### Patterns NOT Used
1. **pytest.skip()**: Never used for empty datasets - always use assertions
2. **Mock objects**: Tests use real datasets and file paths
3. **Resource cleanup**: No explicit cleanup patterns in dataset tests
4. **Complex test classes**: Only plain pytest functions with fixtures
5. **Database/external service mocking**: All data is file-based

### Anti-Patterns to Avoid

```python
# ❌ DON'T - pytest.skip for empty datasets
def test_empty_dataset():
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")

# ✅ DO - Assert-based validation
def test_dataset_validation():
    assert len(dataset) > 0, "Dataset should not be empty"
    # Test normal functionality
    datapoint = dataset[0]
    assert isinstance(datapoint, dict)
    assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}

# Note: Performance timing tests are allowed but not commonly implemented yet

# ✅ DO - Focus on correctness validation
def test_data_correctness():
    datapoint = dataset[0]
    assert isinstance(datapoint, dict)
```

## CI Integration Note

Pylon has basic CI integration that runs pytest with specific flags, but this area is still being developed and should not be considered a mature pattern yet.

## Testing Command Examples

```bash
# Run with sample control (typical development)
pytest tests/data/datasets/pcr_datasets/test_threedmatch_dataset.py --samples=5

# Run full validation
pytest tests/data/datasets/ -v

# Run specific test patterns
pytest tests/data/datasets/ -k "test_validation"
```
