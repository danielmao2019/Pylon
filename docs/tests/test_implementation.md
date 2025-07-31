# Testing Implementation Guidelines

## Critical Rules

**CRITICAL:** Test directories should **NOT** contain `__init__.py` files - they are not Python packages.

**CRITICAL:** Test subdirectories should **NOT** start with `test_` prefix - use descriptive names instead:
- ✅ `tests/utils/automation/progress_tracking/session_progress/`
- ✅ `tests/utils/automation/progress_tracking/base_progress_tracker/`
- ❌ `tests/utils/automation/progress_tracking/test_session_progress/`
- ❌ `tests/utils/automation/progress_tracking/test_base_progress_tracker/`

**CRITICAL:** Use pytest functions only - NO test classes:
- **Framework**: Use `pytest` with plain functions ONLY
- **NO test classes**: Never use `class Test*` - always write `def test_*()` functions
- **Parametrization**: Use `@pytest.mark.parametrize` for multiple test cases instead of test classes
- **Invalid case testing**: Use `pytest.raises()` instead of try-catch blocks for testing expected failures
- **Test organization**: For large test modules, split by test patterns into separate files:

**CRITICAL:** Never use `if __name__ == '__main__':` in test files:
- **Test files are for testing, not for direct execution** - use pytest to run tests
- **Wrong**: `if __name__ == '__main__': test_my_function()` in test files
- **Correct**: Run tests via `pytest tests/test_file.py`
- **Use `if __name__ == '__main__':` only in**: Scripts, main entry points, utilities, demos - never in test files
  ```
  tests/criteria/base_criterion/
  ├── test_initialization.py      # Initialization pattern
  ├── test_buffer_management.py   # Threading/async buffer tests
  ├── test_device_handling.py     # Device transfer tests
  ├── test_edge_cases.py          # Error handling and edge cases
  └── test_determinism.py         # Reproducibility tests
  ```

## Core Testing Principles

1. **Always use pytest functions**: Write `def test_*()` functions, never `class Test*`
2. **Use pytest.mark.parametrize**: When testing multiple similar cases, use parametrization instead of separate functions
3. **Use pytest.raises for invalid cases**: Test expected failures with `pytest.raises()`, not try-catch blocks
4. **Split large test files**: When test files grow large (>300 lines), split by functional aspects
5. **Prefer real tests over mocks**: Only use mocking when absolutely necessary - prefer real object testing
6. **Never use __init__.py in tests**: Test directories are not Python modules
7. **No test_ prefix for directories**: Test subdirectories should use descriptive names, not `test_` prefix
8. **Use conftest.py wisely**: Put common code in conftest.py as fixtures, shared across multiple test files
9. **Cache version discrimination**: Every dataset must include tests verifying version hash discrimination

## Examples of Correct vs Incorrect Test Patterns

### Function vs Class Testing

```python
# ❌ WRONG - Never use test classes
class TestModelName:
    def test_initialization(self):
        model = ModelName()
        assert model is not None

# ✅ CORRECT - Use plain pytest functions  
def test_model_name_initialization():
    model = ModelName()
    assert model is not None
```

### Parametrization vs Multiple Functions

```python
# ❌ WRONG - Multiple similar tests as separate functions
def test_model_batch_size_1():
    test_with_batch_size(1)

def test_model_batch_size_2():
    test_with_batch_size(2)

# ✅ CORRECT - Use parametrize for multiple test cases
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_model_different_batch_sizes(batch_size):
    model = ModelName()
    input_data = generate_dummy_data(batch_size=batch_size)
    output = model(input_data)
    assert output.shape[0] == batch_size
```

### Invalid Case Testing

```python
# ❌ WRONG - Using try-catch for expected failures
def test_invalid_input():
    model = ModelName()
    try:
        model.process(None)  # Should fail
        assert False, "Should have raised an exception"
    except ValueError:
        pass  # Expected

# ✅ CORRECT - Use pytest.raises for expected failures
def test_invalid_input():
    model = ModelName()
    with pytest.raises(ValueError) as exc_info:
        model.process(None)
    assert "input cannot be None" in str(exc_info.value)
```

### Mocking vs Real Testing

```python
# ❌ WRONG - Unnecessary mocking
@patch('some_module.real_function')
def test_with_mock(mock_func):
    mock_func.return_value = "fake_result"
    result = my_function()
    assert result == "processed_fake_result"

# ✅ CORRECT - Use real objects when possible
def test_with_real_objects():
    real_input = create_test_data()
    result = my_function(real_input)
    assert result.shape == expected_shape
    assert result.dtype == torch.float32
```

## Test Organization Strategies

### Small Files (< 300 lines)
Reorder functions to group invalid tests at the bottom:
```python
# ============================================================================
# VALID TESTS - SUCCESSFUL FUNCTIONALITY
# ============================================================================

def test_valid_case_1():
    # Normal functionality test
    pass

def test_valid_case_2():
    # Another valid test
    pass

# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_invalid_input():
    with pytest.raises(AssertionError):
        # Test invalid input handling
        pass
```

### Large Files (> 300 lines)
Split into separate files by functional aspects:
```
tests/module/
├── test_initialization.py   # Object creation and setup
├── test_core_functionality.py  # Main feature testing
├── test_edge_cases.py       # Boundary conditions
├── test_invalid_cases.py    # pytest.raises tests
└── test_integration.py      # End-to-end testing
```

Or split by valid/invalid pattern:
```
tests/module/
├── test_valid_cases.py      # All valid functionality tests
└── test_invalid_cases.py    # All pytest.raises tests
```

### Using conftest.py for Common Code

When you notice code duplication across test files:

```python
# tests/module/conftest.py

@pytest.fixture
def sample_data():
    """Provide common test data used across multiple test files."""
    return {
        'input': torch.randn(2, 3, 64, 64, dtype=torch.float32),
        'labels': torch.randint(0, 10, (2,), dtype=torch.int64)
    }

@pytest.fixture
def ModelFactory():
    """Provide model class factory for consistent test setup."""
    def _create_model(config=None):
        default_config = {'hidden_dim': 128, 'num_classes': 10}
        if config:
            default_config.update(config)
        return SomeModel(**default_config)
    return _create_model

# Usage in test files - no imports needed:
def test_model_forward(ModelFactory, sample_data):
    model = ModelFactory()
    output = model(sample_data['input'])
    assert output.shape == (2, 10)
```

## Testing Focus

**All tests in Pylon are for "your implementation"** - code we've written or integrated:
- **Base classes and wrappers**: Comprehensive testing with all 9 patterns
- **Domain-specific models/losses**: Focus on integration and API correctness
  - Test successful execution with dummy inputs
  - Verify basic input/output shapes and types
  - Test gradient flow and device handling
  - No need to verify mathematical correctness against papers

**Models Module Testing**: We don't do deep testing for the models module, because those are all directly copied over from official implementations of their papers, with API fixes to adjust to Pylon. We just make sure that the API works - i.e., we are able to use those code in Pylon. We don't test for correctness of those code.

**Note**: We do not write separate tests for "official_implementation" - all integrated code is tested as "your implementation".

## Critical Testing Patterns for Pylon Components

**Test Configuration Requirements:**
```python
# ✅ CORRECT - Always use batch_size=1 for validation/evaluation tests
'val_dataloader': {
    'class': torch.utils.data.DataLoader,
    'args': {
        'batch_size': 1,  # REQUIRED for validation/evaluation
        'shuffle': False,
        'collate_fn': {
            'class': BaseCollator,  # REQUIRED - never use default collate_fn
            'args': {},
        },
    }
},

# ❌ WRONG - These patterns will cause test failures
'batch_size': 32,  # Wrong for validation/evaluation
'collate_fn': None,  # Wrong - breaks meta_info handling
```

**Component Testing Dependencies:**
- **Metric classes**: Must have DIRECTIONS attribute before testing
- **Trainer classes**: Initialize metric before early stopping
- **API contracts**: SingleTaskMetric._compute_score receives tensors, not dicts
- **Error fixing**: Fix root causes, don't hide errors with try-except blocks
- **Dataset classes**: Must implement `_get_cache_version_dict()` and include discrimination tests

## Dataset Cache Version Testing

**MANDATORY for all dataset implementations**: Every dataset class must include tests that verify cache version hash discrimination.

### Test Requirements

All dataset tests must include a `test_cache_version_discrimination()` function that verifies:

1. **Same parameters produce same hash**
2. **Different parameters produce different hashes** 
3. **All content-affecting parameters are tested**

### Example Implementation

```python
def test_dataset_cache_version_discrimination():
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
    
    # Test ALL parameters that affect dataset content
    # Add more assertions for every content-affecting parameter
```

### Critical Test Patterns

#### **Test All Content-Affecting Parameters**
```python
# ✅ CORRECT - Test every parameter that changes dataset content
def test_synthetic_dataset_discrimination():
    with tempfile.TemporaryDirectory() as temp_dir:
        base_args = {
            'data_root': temp_dir,
            'dataset_size': 100,
            'rotation_mag': 45.0,
            'translation_mag': 0.5,
            'matching_radius': 0.05,
        }
        
        # Test each parameter individually
        for param_name, new_value in [
            ('dataset_size', 200),
            ('rotation_mag', 30.0), 
            ('translation_mag', 0.3),
            ('matching_radius', 0.1),
        ]:
            dataset1 = MyDataset(**base_args)
            modified_args = base_args.copy()
            modified_args[param_name] = new_value
            dataset2 = MyDataset(**modified_args)
            
            assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash(), \
                f"Parameter {param_name} should affect cache version hash"
```

#### **Test Optional Parameters**
```python
# ✅ CORRECT - Test None vs specified values
def test_optional_parameter_discrimination():
    with tempfile.TemporaryDirectory() as temp_dir:
        # None vs specified should have different hashes
        dataset1 = MyDataset(data_root=temp_dir, camera_count=None)
        dataset2 = MyDataset(data_root=temp_dir, camera_count=10)
        assert dataset1.get_cache_version_hash() != dataset2.get_cache_version_hash()
        
        # Different specified values should have different hashes
        dataset3 = MyDataset(data_root=temp_dir, camera_count=20)
        assert dataset2.get_cache_version_hash() != dataset3.get_cache_version_hash()
```

#### **Test Comprehensive No-Collision**
```python
# ✅ CORRECT - Comprehensive collision detection
def test_comprehensive_no_hash_collisions():
    """Ensure no hash collisions across many different configurations."""
    datasets = []
    
    # Generate many different dataset configurations
    for param1 in [value1, value2, value3]:
        for param2 in [valueA, valueB]:
            datasets.append(MyDataset(param1=param1, param2=param2))
    
    # Collect all hashes
    hashes = [dataset.get_cache_version_hash() for dataset in datasets]
    
    # Ensure all hashes are unique (no collisions)
    assert len(hashes) == len(set(hashes)), \
        f"Hash collision detected! Duplicate hashes found in: {hashes}"
    
    # Ensure all hashes are properly formatted
    for hash_val in hashes:
        assert isinstance(hash_val, str), f"Hash must be string, got {type(hash_val)}"
        assert len(hash_val) == 16, f"Hash must be 16 characters, got {len(hash_val)}"
```

### Test Organization

Place cache version tests in the same test file as other dataset tests:

```
tests/data/datasets/
├── test_cache_version_discrimination.py  # Cross-dataset discrimination tests
└── my_dataset/
    ├── test_initialization.py
    ├── test_data_loading.py
    └── test_cache_version.py              # Dataset-specific version tests
```
