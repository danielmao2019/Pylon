# Testing Guidelines

This directory contains comprehensive testing guidelines for the Pylon framework.

## Important Rules

**CRITICAL:** Test directories should **NOT** contain `__init__.py` files - they are not Python packages.
- This prevents test directories from being treated as importable modules
- Ensures pytest discovery works properly
- Avoids complex import issues when sharing code between test files

**CRITICAL:** Always run pytest from the project root directory. Never run from within test subdirectories.
- Ensures proper module resolution and import paths
- Example: `pytest tests/module/test_file.py` (from root) ✅
- Example: `cd tests/module && pytest test_file.py` (from subdirectory) ❌

## Documentation Structure

- `testing_philosophy.md` - Core testing principles and patterns
- `test_patterns.md` - Detailed test pattern taxonomy and examples  
- `test_implementation.md` - Implementation guidelines and best practices
- `conftest_guidelines.md` - Guidelines for using conftest.py files properly
- `dummy_data.md` - Standardized dummy data generation for tests
