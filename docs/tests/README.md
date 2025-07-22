# Testing Guidelines

This directory contains comprehensive testing guidelines for the Pylon framework.

## Critical Rules

**CRITICAL:** Test directories should **NOT** contain `__init__.py` files - they are not Python packages.
- This prevents test directories from being treated as importable modules
- Ensures pytest discovery works properly
- Avoids complex import issues when sharing code between test files

**CRITICAL:** Always run pytest from the project root directory. Never run from within test subdirectories.
- Ensures proper module resolution and import paths
- Example: `pytest tests/module/test_file.py` (from root) ✅
- Example: `cd tests/module && pytest test_file.py` (from subdirectory) ❌

**CRITICAL:** NEVER import from conftest.py files. Always use fixtures instead.
- Define helper classes as fixtures that return the class definition
- Fixtures are auto-discovered by pytest without imports
- Avoids import complexity in test directories without `__init__.py`

## Core Testing Principles

1. **Always use pytest functions**: Write `def test_*()` functions, never `class Test*`
2. **Use pytest.mark.parametrize**: When testing multiple similar cases, use parametrization
3. **Use pytest.raises for invalid cases**: Test expected failures with `pytest.raises()`, not try-catch
4. **Split large test files**: When files grow >300 lines, split by functional aspects
5. **Prefer real tests over mocks**: Only mock when absolutely necessary
6. **Never use __init__.py in tests**: Test directories are not Python modules
7. **Use conftest.py wisely**: Put common code as fixtures, shared across multiple test files

## Documentation Structure

- `testing_philosophy.md` - Core testing principles and 9 test patterns
- `test_implementation.md` - Implementation guidelines and best practices
- `conftest_guidelines.md` - Guidelines for using conftest.py files properly (fixtures only!)
- `dummy_data.md` - Optional dummy data generation for tests

**For tensor type requirements, see `@CLAUDE.md` Section 4.1 "Tensor Type Assumptions".**
