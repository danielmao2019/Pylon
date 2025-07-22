"""
Test BaseProgressTracker functionality - INVALID CASES (pytest.raises).

Following CLAUDE.md testing patterns:
- Invalid input testing with exception verification
- Abstract method contract testing
"""
import pytest
from typing import List, Literal
from utils.automation.progress_tracking.base_progress_tracker import BaseProgressTracker, ProgressInfo


# ============================================================================
# CONCRETE TEST IMPLEMENTATION FOR TESTING ABSTRACT CLASS
# ============================================================================

class ConcreteProgressTracker(BaseProgressTracker):
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
        """Mock implementation for testing."""
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


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_base_progress_tracker_input_validation():
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