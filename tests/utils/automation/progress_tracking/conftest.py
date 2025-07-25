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
def ConcreteProgressTracker():
    """Fixture that provides ConcreteProgressTracker class for testing BaseProgressTracker."""
    from typing import List, Literal
    from utils.automation.progress_tracking.base_progress_tracker import BaseProgressTracker, ProgressInfo
    
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
        
        def calculate_progress(self, force_progress_recompute: bool = False) -> ProgressInfo:
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
    
    return ConcreteProgressTrackerImpl
