"""
Test EvaluatorTracker functionality - INVALID CASES (pytest.raises).

Following CLAUDE.md testing patterns:
- Invalid input testing with exception verification
"""
import pytest
from agents.tracker.evaluator_tracker import EvaluatorTracker


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_evaluator_tracker_nonexistent_work_dir():
    """Test initialization with nonexistent work directory."""
    nonexistent_dir = "/this/path/does/not/exist"
    
    with pytest.raises(AssertionError) as exc_info:
        EvaluatorTracker(nonexistent_dir)
    
    assert "work_dir does not exist" in str(exc_info.value)


def test_evaluator_tracker_invalid_work_dir_type():
    """Test initialization with invalid work_dir type."""
    with pytest.raises(AssertionError) as exc_info:
        EvaluatorTracker(123)  # Integer instead of string
    
    assert "work_dir must be str" in str(exc_info.value)
