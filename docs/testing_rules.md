# Critical Testing Rules to Remember

## ❌ NEVER DO: `if __name__ == '__main__':` in Test Files

**RULE**: Never include `if __name__ == '__main__':` blocks in test files.

**Why this is wrong:**
- Test files should only contain test functions and helper code
- Tests should be run via pytest, not directly as scripts
- `if __name__ == '__main__':` blocks in test files indicate improper test design
- It makes tests harder to discover and run through proper test runners

**❌ Wrong:**
```python
# test_something.py
def test_my_function():
    assert my_function() == expected_result

if __name__ == '__main__':  # NEVER DO THIS IN TEST FILES
    test_my_function()
    print("✓ Test passed")
```

**✅ Correct:**
```python
# test_something.py
def test_my_function():
    assert my_function() == expected_result

# No main block needed - pytest will discover and run this automatically
```

**How to run tests properly:**
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_something.py

# Run specific test function
pytest tests/test_something.py::test_my_function
```

## Other Testing Reminders

- Use `def test_*()` function names for pytest discovery
- Use descriptive test function names that explain what is being tested
- Group related tests in the same file
- Use fixtures for setup/teardown, not manual setup in test functions
- Test files should NOT contain `__init__.py` files (they are not Python packages)

## When `if __name__ == '__main__':` IS Appropriate

- **Script files**: Files meant to be executed directly
- **Main entry points**: Application startup files
- **Utility scripts**: Standalone tools and utilities
- **Examples/demos**: Files demonstrating usage

**✅ Appropriate usage:**
```python
# main.py or some_script.py
def main():
    # Application logic here
    pass

if __name__ == '__main__':  # Appropriate for scripts
    main()
```

## Summary

**Remember: Test files are for testing, not for direct execution. Use pytest to run tests, never include main blocks in test files.**