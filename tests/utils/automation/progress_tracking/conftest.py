"""
Shared test fixtures specific to progress tracking tests.

This conftest.py provides fixtures specific to progress tracking
that aren't needed in other automation test modules.
"""
import pytest


# ============================================================================
# PROGRESS TRACKING SPECIFIC TEST HELPER CLASSES AS FIXTURES  
# ============================================================================

@pytest.fixture
def TrainerProgressTrackerForTesting():
    """Fixture that provides TrainerProgressTracker for testing BaseProgressTracker functionality."""
    from utils.automation.progress_tracking.trainer_progress_tracker import TrainerProgressTracker
    return TrainerProgressTracker


@pytest.fixture 
def EvaluatorProgressTrackerForTesting():
    """Fixture that provides EvaluatorProgressTracker for testing BaseProgressTracker functionality."""
    from utils.automation.progress_tracking.evaluator_progress_tracker import EvaluatorProgressTracker
    return EvaluatorProgressTracker
