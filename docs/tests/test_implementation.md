# Testing Implementation Guidelines

## Critical Rules

**CRITICAL:** Test directories should **NOT** contain `__init__.py` files - they are not Python packages.

**CRITICAL:** Use pytest functions only - NO test classes:
- **Framework**: Use `pytest` with plain functions ONLY
- **NO test classes**: Never use `class Test*` - always write `def test_*()` functions
- **Parametrization**: Use `@pytest.mark.parametrize` for multiple test cases instead of test classes
- **Invalid case testing**: Use `pytest.raises()` instead of try-catch blocks for testing expected failures
- **Test organization**: For large test modules, split by test patterns into separate files:
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
7. **Use conftest.py wisely**: Put common code in conftest.py as fixtures, shared across multiple test files

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
