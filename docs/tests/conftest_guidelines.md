# conftest.py Guidelines

## Core Principles

**CRITICAL RULE: NEVER import from conftest.py files. Always use fixtures instead.**

**conftest.py files are for pytest fixtures ONLY, not for imports**

## What conftest.py IS for:
- **Fixtures**: Functions decorated with `@pytest.fixture` that provide test data or setup
- **Test helper class factories**: Fixtures that return class definitions for test helpers
- **Common test setup/teardown**: Shared test environment preparation

## What conftest.py is NOT for:
- Classes or functions that you import directly
- General utility functions that aren't fixtures
- Source code that belongs in the main codebase

## NEVER Import from conftest.py:
- **NEVER do**: `from .conftest import SomeClass`
- **ALWAYS do**: Define `SomeClass` as a fixture and use it as a parameter
- **Best practice**: If you need a class, create a fixture that returns the class definition

## Auto-discovery Rules:
- Only fixtures (functions decorated with `@pytest.fixture`) are auto-discovered
- Fixtures are automatically available to test functions without imports
- Test functions receive fixtures by adding them as parameters

## When to Use conftest.py for Test Helper Classes:

### ✅ DO use conftest.py when:
- The same helper class is used across multiple test files **within the same directory tree**
- The class is test-specific and not part of the main codebase
- You want to avoid code duplication across test files
- **ALWAYS define as fixtures, never as direct imports**

### ❌ DON'T use conftest.py when:
- The class is only used in one test file (define it directly in that file)
- The class belongs in the actual source modules
- You're trying to import classes directly (use fixtures instead)

### Fixture Approach Benefits:
Since test directories do NOT contain `__init__.py` files, using fixtures instead of imports avoids all import complexity. **Fixtures are auto-discovered by pytest regardless of directory structure, making them the ideal solution for sharing code between test files.**

## Examples

### ✅ CORRECT - Helper class as fixture in conftest.py:
```python
# tests/utils/automation/progress_tracking/conftest.py

@pytest.fixture
def ConcreteProgressTracker():
    """Fixture that provides ConcreteProgressTracker class for testing BaseProgressTracker."""
    from typing import List, Literal
    from agents.tracker.base_progress_tracker import BaseProgressTracker, ProgressInfo
    
    class ConcreteProgressTrackerImpl(BaseProgressTracker):
        """Concrete implementation for testing BaseProgressTracker functionality."""
        
        def __init__(self, work_dir: str, config=None, runner_type: Literal['trainer', 'evaluator'] = 'trainer'):
            super().__init__(work_dir, config)
            self._runner_type = runner_type
            self._test_progress_result = None
        
        def get_runner_type(self) -> Literal['trainer', 'evaluator']:
            return self._runner_type
        
        def get_expected_files(self) -> List[str]:
            return ["test_file.json", "another_file.pt"]
        
        def get_log_pattern(self) -> str:
            return "test_*.log"
        
        def calculate_progress(self) -> ProgressInfo:
            # Mock implementation for testing
            if self._test_progress_result is None:
                return ProgressInfo(
                    completed_epochs=10,
                    progress_percentage=50.0,
                    early_stopped=False,
                    early_stopped_at_epoch=None,
                    runner_type=self._runner_type,
                    total_epochs=20
                )
            return self._test_progress_result
        
        def set_test_progress_result(self, result: ProgressInfo):
            """Test helper to control calculate_progress output."""
            self._test_progress_result = result
    
    return ConcreteProgressTrackerImpl

# Usage in test files - auto-discovered, no import needed:
def test_something(ConcreteProgressTracker):
    tracker = ConcreteProgressTracker("/some/work/dir")
    # test implementation
```

### ✅ CORRECT - Data fixture in conftest.py:
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

### ❌ WRONG - Importing from conftest.py:
```python
# ❌ NEVER DO THIS:
from .conftest import ConcreteProgressTracker  # WRONG - no imports from conftest!
from ..conftest import ConcreteProgressTracker  # WRONG - no imports from conftest!

def test_something():
    tracker = ConcreteProgressTracker("/some/work/dir")  # WRONG approach
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

When multiple test files share the same helper classes via fixtures:

```
tests/utils/automation/progress_tracking/
├── conftest.py                           # Shared fixtures (including class factories)
├── test_runner_detection.py             # Uses fixtures from conftest.py
├── test_progress_tracker_factory/
│   ├── test_valid_cases.py              # Uses fixtures from ../conftest.py  
│   └── test_invalid_cases.py            # Uses fixtures from ../conftest.py
└── test_base_progress_tracker/
    ├── test_valid_cases.py              # Uses fixtures from ../conftest.py
    └── test_invalid_cases.py            # Uses fixtures from ../conftest.py
```

## Fixture Usage Patterns

```python
# ✅ CORRECT - Fixtures are auto-discovered, no imports needed:
def test_something(ConcreteProgressTracker, temp_work_dir, create_epoch_files):
    # All fixtures available automatically as parameters
    tracker = ConcreteProgressTracker(temp_work_dir)
    create_epoch_files(temp_work_dir, 0)
    # test implementation
```

## Key Benefits of Fixture Approach

1. **No import complexity**: Fixtures work regardless of directory structure
2. **Auto-discovery**: pytest automatically finds and provides fixtures
3. **No __init__.py needed**: Avoids module structure requirements
4. **Cleaner test code**: Test functions simply list needed fixtures as parameters
5. **Better error messages**: pytest clearly shows missing fixtures vs import errors
