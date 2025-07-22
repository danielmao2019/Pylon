# Testing Implementation Guidelines

## Critical Rules

**CRITICAL:** Test directories should **NOT** contain `__init__.py` files - they are not Python packages.

**CRITICAL:** Use pytest functions only - NO test classes:
- **Framework**: Use `pytest` with plain functions ONLY
- **NO test classes**: Never use `class Test*` - always write `def test_*()` functions
- **Parametrization**: Use `@pytest.mark.parametrize` for multiple test cases instead of test classes
- **Test organization**: For large test modules, split by test patterns into separate files:
  ```
  tests/criteria/base_criterion/
  ├── test_initialization.py      # Initialization pattern
  ├── test_buffer_management.py   # Threading/async buffer tests
  ├── test_device_handling.py     # Device transfer tests
  ├── test_edge_cases.py          # Error handling and edge cases
  └── test_determinism.py         # Reproducibility tests
  ```

## Examples of Correct vs Incorrect Test Patterns

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
Split into separate files within subdirectories:
```
tests/module/
├── test_valid_cases.py      # All valid functionality tests
└── test_invalid_cases.py    # All pytest.raises tests
```

## Testing Focus

**All tests in Pylon are for "your implementation"** - code we've written or integrated:
- **Base classes and wrappers**: Comprehensive testing with all 9 patterns
- **Domain-specific models/losses**: Focus on integration and API correctness
  - Test successful execution with dummy inputs
  - Verify basic input/output shapes and types
  - Test gradient flow and device handling
  - No need to verify mathematical correctness against papers

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
