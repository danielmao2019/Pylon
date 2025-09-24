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
    """Fixture that provides TrainerTracker for testing BaseTracker functionality."""
    from agents.tracker.trainer_tracker import TrainerTracker
    return TrainerTracker


@pytest.fixture 
def EvaluatorProgressTrackerForTesting():
    """Fixture that provides EvaluatorTracker for testing BaseTracker functionality."""
    from agents.tracker.evaluator_tracker import EvaluatorTracker
    return EvaluatorTracker
