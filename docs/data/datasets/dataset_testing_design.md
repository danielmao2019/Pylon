# Dataset Testing Design in Pylon

## Overview

This document outlines testing strategies and patterns for dataset implementations in Pylon, ensuring robustness, correctness, and integration with the broader framework.

## Testing Philosophy

### Comprehensive Validation

Dataset tests should validate:
1. **Data Structure**: Correct tensor shapes, types, and dictionary structure
2. **Framework Integration**: Compatibility with transforms, dataloaders, and collators
3. **Performance**: Memory usage and loading speed
4. **Determinism**: Reproducible behavior across runs
5. **Edge Cases**: Empty datasets, boundary conditions, error handling

### Test Organization

```
tests/data/datasets/your_domain/
├── test_your_dataset.py              # Main functionality tests
├── test_your_dataset_determinism.py  # Reproducibility tests (optional)
├── test_your_dataset_performance.py  # Performance tests (optional)
└── conftest.py                       # Shared test fixtures
```

## Core Testing Patterns

### 1. Data Structure Validation

Create comprehensive validation functions for each data type:

```python
def validate_inputs(inputs: Dict[str, Any]) -> None:
    """Validate input dictionary structure and types."""
    assert isinstance(inputs, dict), f"Expected dict, got {type(inputs)}"
    
    # Check expected keys
    expected_keys = {'input1', 'input2'}  # Based on your dataset
    assert inputs.keys() == expected_keys, f"Keys mismatch: {inputs.keys()}"
    
    # Validate each input
    for key, tensor in inputs.items():
        assert isinstance(tensor, torch.Tensor), f"{key} not a tensor: {type(tensor)}"
        assert tensor.dtype == torch.float32, f"{key} wrong dtype: {tensor.dtype}"
        # Add shape and range validations specific to your data

def validate_labels(labels: Dict[str, Any]) -> None:
    """Validate label dictionary structure and types."""
    assert isinstance(labels, dict), f"Expected dict, got {type(labels)}"
    
    expected_keys = {'label1'}  # Based on your dataset
    assert labels.keys() == expected_keys, f"Keys mismatch: {labels.keys()}"
    
    # Validate each label
    for key, tensor in labels.items():
        assert isinstance(tensor, torch.Tensor), f"{key} not a tensor: {type(tensor)}"
        # Add specific validations for your labels

def validate_meta_info(meta_info: Dict[str, Any], expected_idx: int) -> None:
    """Validate meta_info dictionary."""
    assert isinstance(meta_info, dict), f"Expected dict, got {type(meta_info)}"
    
    # Check required fields
    assert 'idx' in meta_info, "meta_info missing 'idx'"
    assert meta_info['idx'] == expected_idx, f"idx mismatch: {meta_info['idx']} != {expected_idx}"
    
    # Validate other expected fields
    for key in ['file_path', 'scene_name']:  # Adjust for your dataset
        assert key in meta_info, f"meta_info missing '{key}'"
        assert isinstance(meta_info[key], str), f"{key} not string: {type(meta_info[key])}"
```

### 2. Dataset Fixtures

Use pytest fixtures for test setup:

```python
# conftest.py
import pytest
from your_dataset import YourDataset

@pytest.fixture(params=['train', 'val', 'test'])
def dataset(request):
    """Fixture for creating dataset instances."""
    split = request.param
    return YourDataset(
        data_root='./path/to/test/data',
        split=split,
        # Add other parameters
    )

@pytest.fixture
def single_datapoint(dataset):
    """Fixture for getting a single datapoint."""
    if len(dataset) > 0:
        return dataset[0]
    else:
        pytest.skip("Dataset is empty")
```

### 3. Parametrized Testing

Test across different splits and configurations:

```python
@pytest.mark.parametrize('dataset', ['train', 'val', 'test'], indirect=True)
def test_dataset_structure(dataset):
    """Test basic dataset structure across all splits."""
    
    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        assert isinstance(datapoint, dict)
        assert datapoint.keys() == {'inputs', 'labels', 'meta_info'}
        
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)
    
    # Test multiple samples
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    # Test first, last, and random samples
    indices_to_test = [0]
    if len(dataset) > 1:
        indices_to_test.append(len(dataset) - 1)
    if len(dataset) > 2:
        indices_to_test.append(len(dataset) // 2)
    
    for idx in indices_to_test:
        validate_datapoint(idx)
```

### 4. Parallel Testing for Performance

Use threading to test multiple samples efficiently:

```python
from concurrent.futures import ThreadPoolExecutor
import random

def test_dataset_samples_parallel(dataset, max_samples=10):
    """Test multiple dataset samples in parallel."""
    
    def validate_datapoint(idx: int) -> None:
        datapoint = dataset[idx]
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
        validate_meta_info(datapoint['meta_info'], idx)
    
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    # Test random samples
    num_samples = min(len(dataset), max_samples)
    indices = random.sample(range(len(dataset)), num_samples)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(validate_datapoint, indices))
```

### 5. Command Line Test Control

Support command line options for test configuration:

```python
# conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--samples",
        action="store",
        type=int,
        default=None,
        help="Number of samples to test"
    )

@pytest.fixture
def max_samples(request):
    """Get max samples from command line."""
    return request.config.getoption("--samples")

@pytest.fixture
def get_samples_to_test():
    """Helper to determine how many samples to test."""
    def _get_samples(dataset_size: int, max_samples: Optional[int], default: int = 5) -> int:
        if max_samples is not None:
            return min(dataset_size, max_samples)
        return min(dataset_size, default)
    return _get_samples

# Usage in tests
def test_dataset_functionality(dataset, max_samples, get_samples_to_test):
    num_samples = get_samples_to_test(len(dataset), max_samples, default=5)
    # Test num_samples datapoints
```

## Specific Testing Scenarios

### 1. Transform Compatibility

Test that your dataset works with Pylon transforms:

```python
def test_dataset_with_transforms(dataset):
    """Test dataset compatibility with transforms."""
    from data.transforms import SomeTransform  # Import relevant transforms
    
    # Apply transform
    dataset.set_transforms(SomeTransform())
    
    # Test that transformed data is still valid
    if len(dataset) > 0:
        datapoint = dataset[0]
        validate_inputs(datapoint['inputs'])
        validate_labels(datapoint['labels'])
```

### 2. DataLoader Integration

Test integration with PyTorch DataLoader:

```python
def test_dataset_with_dataloader(dataset):
    """Test dataset with PyTorch DataLoader."""
    from torch.utils.data import DataLoader
    from data.collators import SomeCollator  # Import appropriate collator
    
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=2,
        collate_fn=SomeCollator(),
    )
    
    # Test first batch
    batch = next(iter(dataloader))
    assert isinstance(batch, dict)
    assert 'inputs' in batch and 'labels' in batch
    
    # Validate batch structure
    inputs = batch['inputs']
    labels = batch['labels']
    
    # Check batch dimensions
    for key, tensor in inputs.items():
        assert tensor.shape[0] <= 2, f"Batch size too large for {key}: {tensor.shape}"
```

### 3. Memory Usage Testing

Monitor memory consumption:

```python
import psutil
import gc

def test_memory_usage(dataset):
    """Test that dataset doesn't leak memory."""
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Load multiple datapoints
    for i in range(min(10, len(dataset))):
        datapoint = dataset[i]
        del datapoint
        gc.collect()
    
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    
    # Allow some memory increase but not excessive
    max_allowed_mb = 100  # Adjust based on your dataset
    assert memory_increase < max_allowed_mb * 1024 * 1024, \
        f"Memory increased by {memory_increase / 1024 / 1024:.1f} MB"
```

### 4. Determinism Testing

Test reproducible behavior:

```python
def test_dataset_determinism(dataset):
    """Test that dataset produces deterministic results."""
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    # Load same datapoint multiple times
    idx = 0
    datapoint1 = dataset[idx]
    datapoint2 = dataset[idx]
    
    # Check that inputs are identical
    for key in datapoint1['inputs']:
        tensor1 = datapoint1['inputs'][key]
        tensor2 = datapoint2['inputs'][key]
        assert torch.equal(tensor1, tensor2), f"Non-deterministic behavior in {key}"
    
    # Check labels
    for key in datapoint1['labels']:
        tensor1 = datapoint1['labels'][key]
        tensor2 = datapoint2['labels'][key]
        assert torch.equal(tensor1, tensor2), f"Non-deterministic behavior in {key}"
```

### 5. Edge Case Testing

Test boundary conditions and error cases:

```python
def test_dataset_edge_cases(dataset):
    """Test edge cases and boundary conditions."""
    
    # Test empty dataset handling
    if len(dataset) == 0:
        pytest.skip("Testing empty dataset behavior")
        # Could test that len(dataset) == 0 is handled gracefully
    
    # Test first and last indices
    if len(dataset) > 0:
        first = dataset[0]
        last = dataset[len(dataset) - 1]
        
        # Validate both are properly formatted
        validate_inputs(first['inputs'])
        validate_inputs(last['inputs'])

def test_invalid_indices(dataset):
    """Test error handling for invalid indices."""
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    # Test negative index (should work in Python)
    datapoint = dataset[-1]
    validate_inputs(datapoint['inputs'])
    
    # Test out of bounds index
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]
```

## Performance Testing

### Timing Tests

```python
import time

def test_loading_performance(dataset):
    """Test dataset loading performance."""
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    # Time loading of multiple samples
    num_samples = min(10, len(dataset))
    start_time = time.time()
    
    for i in range(num_samples):
        datapoint = dataset[i]
        del datapoint  # Free memory
    
    elapsed = time.time() - start_time
    avg_time = elapsed / num_samples
    
    # Set reasonable performance expectations
    max_time_per_sample = 1.0  # seconds - adjust for your dataset
    assert avg_time < max_time_per_sample, \
        f"Loading too slow: {avg_time:.3f}s per sample"
```

### Cache Effectiveness Testing

If your dataset uses caching:

```python
def test_cache_effectiveness(dataset):
    """Test that caching improves performance."""
    if len(dataset) == 0:
        pytest.skip("Dataset is empty")
    
    # Clear any existing cache
    # (Implementation depends on your caching strategy)
    
    # Time first load (should populate cache)
    start_time = time.time()
    datapoint1 = dataset[0]
    first_load_time = time.time() - start_time
    
    # Time second load (should use cache)
    start_time = time.time()
    datapoint2 = dataset[0]
    second_load_time = time.time() - start_time
    
    # Cache should make second load faster
    assert second_load_time < first_load_time, \
        "Cache not improving performance"
    
    # Results should be identical
    # (Add comparison logic based on your data types)
```

## Test Configuration

### Running Tests

```bash
# Run all dataset tests
pytest tests/data/datasets/your_domain/

# Run with limited samples for faster testing
pytest tests/data/datasets/your_domain/ --samples 3

# Run with verbose output
pytest tests/data/datasets/your_domain/ -v

# Run specific test patterns
pytest tests/data/datasets/your_domain/ -k "test_structure"
```

### Continuous Integration

For CI/CD pipelines, use fast test configurations:

```yaml
# .github/workflows/test.yml
- name: Test datasets
  run: pytest tests/data/datasets/ --samples 2 --tb=short
```

## Best Practices

### 1. Test Data Management

- **Use small test datasets**: Don't require full dataset downloads for tests
- **Create synthetic data**: Generate minimal test data that covers edge cases
- **Mock expensive operations**: Use pytest.mock for slow operations during testing

### 2. Assertion Messages

Provide clear, actionable error messages:

```python
# ✅ GOOD - Clear diagnostic information
assert tensor.shape == expected_shape, \
    f"Shape mismatch for {key}: got {tensor.shape}, expected {expected_shape}"

# ❌ BAD - Uninformative
assert tensor.shape == expected_shape
```

### 3. Test Independence

Ensure tests don't depend on each other:

```python
# ✅ GOOD - Each test is independent
def test_first_sample(dataset):
    datapoint = dataset[0]
    # ... test logic

def test_last_sample(dataset):
    datapoint = dataset[-1]
    # ... test logic

# ❌ BAD - Tests depend on execution order
def test_load_and_cache(dataset):
    # This test might fail if run in isolation
    assert cache_exists()
```

### 4. Resource Cleanup

Clean up resources to prevent test interference:

```python
def test_with_cleanup(dataset):
    try:
        # Test logic that creates resources
        pass
    finally:
        # Always clean up
        cleanup_resources()
```

## Common Testing Pitfalls

1. **Assuming dataset size**: Always check `len(dataset) > 0` before indexing
2. **Hardcoded paths**: Use configurable paths for test data
3. **Missing edge cases**: Test empty datasets, single-item datasets, boundary indices
4. **Ignoring performance**: Don't let tests become unreasonably slow
5. **Incomplete validation**: Validate all aspects of the data structure, not just shapes
6. **Thread safety issues**: Use proper synchronization when testing with multiple workers
