# conftest.py Guidelines

## Core Principles

CRITICAL RULE: NEVER import from conftest.py files. Always use fixtures instead.

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

### ✅ CORRECT - Helper class as fixture in conftest.py (no imports)
```python
# tests/utils/automation/progress_tracking/conftest.py

@pytest.fixture
def ConcreteTracker():
    """Fixture that provides ConcreteTracker class for testing Base tracker API."""
    from typing import List, Literal
    # Provide a simple helper class purely for tests
    class Helper:
        def __init__(self, value: int):
            self.value = value
        def add(self, x: int) -> int:
            return self.value + x
    return Helper

# Usage in test files - auto-discovered, no import needed:
def test_something(Helper):
    inst = Helper(2)
    assert inst.add(3) == 5
```

### ✅ CORRECT - Data fixture in conftest.py (no imports)
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

### ❌ WRONG - Importing from conftest.py
```python
# ❌ NEVER DO THIS:
from .conftest import SomeFixture  # WRONG - no imports from conftest!
from ..conftest import AnotherFixture  # WRONG - no imports from conftest!

def test_something():
    use = SomeFixture  # WRONG approach
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
├── runner_detection/
│   ├── test_valid_cases.py              # Uses fixtures from ../conftest.py  
│   └── test_invalid_cases.py            # Uses fixtures from ../conftest.py
└── some_other_suite/
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
