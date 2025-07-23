"""
Test BaseProgressTracker functionality - INVALID CASES (pytest.raises).

Following CLAUDE.md testing patterns:
- Invalid input testing with exception verification
- Abstract method contract testing
"""
import pytest
from utils.automation.progress_tracking.base_progress_tracker import BaseProgressTracker


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_base_progress_tracker_input_validation(ConcreteProgressTracker):
    """Test input validation during initialization."""
    # Test invalid work_dir type
    with pytest.raises(AssertionError) as exc_info:
        ConcreteProgressTracker(123)  # Integer instead of string
    assert "work_dir must be str" in str(exc_info.value)
    
    # Test nonexistent work_dir
    nonexistent_dir = "/this/path/does/not/exist"
    with pytest.raises(AssertionError) as exc_info:
        ConcreteProgressTracker(nonexistent_dir)
    assert "work_dir does not exist" in str(exc_info.value)


def test_base_progress_tracker_abstract_methods_must_be_implemented():
    """Test that abstract methods must be implemented by subclasses."""
    
    # Test that we can't instantiate BaseProgressTracker directly
    with pytest.raises(TypeError):
        BaseProgressTracker("/tmp")  # Should fail - abstract class
    
    # Test that incomplete implementation fails
    class IncompleteTracker(BaseProgressTracker):
        def get_runner_type(self):
            return 'trainer'
        # Missing other abstract methods
    
    with pytest.raises(TypeError):
        IncompleteTracker("/tmp")  # Should fail - missing abstract methods
