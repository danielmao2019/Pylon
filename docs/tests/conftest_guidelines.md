# conftest.py Guidelines

## Core Principles

**conftest.py files are for pytest fixtures ONLY, not general class definitions**

## What conftest.py IS for:
- **Fixtures**: Functions decorated with `@pytest.fixture` that provide test data or setup
- **Shared test utilities**: Helper functions that multiple test files need
- **Common test helper classes**: When the same helper class is used across multiple test files

## What conftest.py is NOT for:
- General class definitions that aren't test-specific
- Classes that are only used in one test file
- Source code that belongs in the main codebase

## Auto-discovery Rules:
- Only fixtures (functions decorated with `@pytest.fixture`) are auto-discovered
- Classes defined in conftest.py are **NOT** automatically available in test files
- You must explicitly import classes from conftest.py if you need them

## When to Use conftest.py for Test Helper Classes:

### ✅ DO use conftest.py when:
- The same helper class is used in multiple test files
- The class is test-specific and not part of the main codebase
- The class needs setup/teardown functionality

### ❌ DON'T use conftest.py when:
- The class is only used in one test file (define it directly in that file)
- The class belongs in the actual source modules
- You're just trying to avoid imports

## Examples

### ✅ CORRECT - Common helper class in conftest.py:
```python
# tests/utils/automation/progress_tracking/conftest.py

class ConcreteProgressTracker(BaseProgressTracker):
    """Concrete implementation for testing BaseProgressTracker functionality.
    
    Used across multiple test files: test_base_progress_tracker/, test_factory/, etc.
    """
    
    def __init__(self, work_dir: str, config=None, runner_type='trainer'):
        super().__init__(work_dir, config)
        self._runner_type = runner_type
        self._test_progress_result = None
    
    def get_runner_type(self):
        return self._runner_type
    
    def get_expected_files(self):
        return ["test_file.json", "another_file.pt"]
    
    def calculate_progress(self):
        # Mock implementation
        return self._test_progress_result or default_progress_info

# Usage in test files:
from .conftest import ConcreteProgressTracker  # Explicit import needed
```

### ✅ CORRECT - Fixture in conftest.py:
```python
# tests/utils/automation/progress_tracking/conftest.py

@pytest.fixture
def temp_work_dir():
    """Provide a temporary work directory for tests."""
    with tempfile.TemporaryDirectory() as work_dir:
        yield work_dir

@pytest.fixture  
def create_epoch_files():
    """Factory function to create epoch files in work directory."""
    def _create_files(work_dir, epoch_idx, validation_score=0.5):
        # Implementation to create test files
        pass
    return _create_files

# Usage in test files - auto-discovered, no import needed:
def test_something(temp_work_dir, create_epoch_files):
    create_epoch_files(temp_work_dir, 0)
    # test implementation
```

### ❌ WRONG - Class only used in one file:
```python
# Don't put in conftest.py if only used in one test file
# Instead, define directly in the test file:

# tests/utils/automation/progress_tracking/test_specific_functionality.py

class SpecificTestHelper:
    """Helper class only used in this test file."""
    pass

def test_something():
    helper = SpecificTestHelper()
    # test implementation
```

## File Organization

When multiple test files share the same helper class, organize like this:

```
tests/utils/automation/progress_tracking/
├── conftest.py                           # Shared helper classes and fixtures
├── test_runner_detection.py             # Uses helpers from conftest
├── test_progress_tracker_factory/
│   ├── test_valid_cases.py              # Uses helpers from ../conftest.py  
│   └── test_invalid_cases.py            # Uses helpers from ../conftest.py
└── test_base_progress_tracker/
    ├── test_valid_cases.py              # Uses helpers from ../conftest.py
    └── test_invalid_cases.py            # Uses helpers from ../conftest.py
```

## Import Patterns

```python
# From same directory conftest.py:
from .conftest import ConcreteProgressTracker

# From parent directory conftest.py:
from ..conftest import ConcreteProgressTracker

# Fixtures are auto-discovered (no import needed):
def test_something(temp_work_dir, create_epoch_files):
    # fixtures available automatically
    pass
```
