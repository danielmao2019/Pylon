"""
Test BaseProgressTracker functionality and abstract interface contracts - VALID CASES.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Initialization testing pattern
- Determinism testing
- Integration testing
"""
import os
import tempfile
import json
import time
import pytest
from utils.automation.progress_tracking.base_progress_tracker import BaseProgressTracker, ProgressInfo


# ============================================================================
# TESTS FOR BaseProgressTracker - INITIALIZATION
# ============================================================================

def test_base_progress_tracker_initialization_with_config(ConcreteProgressTracker):
    """Test BaseProgressTracker initialization with config."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'model': 'test_model'}
        tracker = ConcreteProgressTracker(work_dir, config)
        
        assert tracker.work_dir == work_dir
        assert tracker.config == config
        assert tracker._cache is None
        assert tracker._cache_time is None


def test_base_progress_tracker_initialization_without_config(ConcreteProgressTracker):
    """Test BaseProgressTracker initialization without config."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        assert tracker.work_dir == work_dir
        assert tracker.config is None
        assert tracker._cache is None
        assert tracker._cache_time is None


# ============================================================================
# TESTS FOR BaseProgressTracker - CACHING MECHANISM
# ============================================================================

def test_base_progress_tracker_caching_basic(ConcreteProgressTracker):
    """Test basic caching functionality."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        # First call should calculate and cache
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 10
        assert tracker._cache is not None
        assert tracker._cache_time is not None
        
        # Second call should return cached result
        progress2 = tracker.get_progress()
        assert progress2 == progress1


def test_base_progress_tracker_force_refresh(ConcreteProgressTracker):
    """Test force refresh bypasses cache."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        # First call - establishes cache
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 10
        
        # Change the mock result
        new_result = ProgressInfo(
            completed_epochs=20,
            progress_percentage=100.0,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='trainer',
            total_epochs=20
        )
        tracker.set_test_progress_result(new_result)
        
        # Normal call should still return cached result
        progress2 = tracker.get_progress()
        assert progress2.completed_epochs == 10  # Still cached
        
        # Force refresh should get new result
        progress3 = tracker.get_progress(force_refresh=True)
        assert progress3.completed_epochs == 20  # New result


def test_base_progress_tracker_cache_timeout(ConcreteProgressTracker):
    """Test that cache expires after timeout."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Use very short cache timeout for testing
        tracker = ConcreteProgressTracker(work_dir)
        tracker._cache_timeout = 0.1  # 100ms
        
        # First call
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 10
        
        # Change mock result
        new_result = ProgressInfo(
            completed_epochs=30,
            progress_percentage=75.0,
            early_stopped=False,
            early_stopped_at_epoch=None,
            runner_type='trainer',
            total_epochs=40
        )
        tracker.set_test_progress_result(new_result)
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Should get new result due to cache timeout
        progress2 = tracker.get_progress()
        assert progress2.completed_epochs == 30


# ============================================================================
# TESTS FOR BaseProgressTracker - PROGRESS.JSON HANDLING
# ============================================================================

def test_base_progress_tracker_saves_progress_json(ConcreteProgressTracker):
    """Test that get_progress saves progress.json file."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        # Get progress
        progress = tracker.get_progress()
        
        # Check that progress.json was created
        progress_file = os.path.join(work_dir, "progress.json")
        assert os.path.exists(progress_file)
        
        # Verify content matches
        with open(progress_file, 'r') as f:
            saved_progress = json.load(f)
        
        assert saved_progress['completed_epochs'] == progress.completed_epochs
        assert saved_progress['progress_percentage'] == progress.progress_percentage
        assert saved_progress['early_stopped'] == progress.early_stopped
        assert saved_progress['early_stopped_at_epoch'] == progress.early_stopped_at_epoch
        assert saved_progress['runner_type'] == progress.runner_type
        assert saved_progress['total_epochs'] == progress.total_epochs


def test_base_progress_tracker_handles_json_save_errors(ConcreteProgressTracker):
    """Test graceful handling of JSON save errors."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Make work_dir readonly to cause save error
        os.chmod(work_dir, 0o444)  # Read-only
        
        try:
            tracker = ConcreteProgressTracker(work_dir)
            # Should not crash even if can't save progress.json
            progress = tracker.get_progress()
            assert progress.completed_epochs == 10
        finally:
            # Restore permissions for cleanup
            os.chmod(work_dir, 0o755)


# ============================================================================
# TESTS FOR BaseProgressTracker - ABSTRACT METHOD CONTRACTS
# ============================================================================

def test_base_progress_tracker_concrete_implementation_contracts(ConcreteProgressTracker):
    """Test that concrete implementation follows expected contracts."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        # Test abstract method implementations
        assert tracker.get_runner_type() in ['trainer', 'evaluator']
        assert isinstance(tracker.get_expected_files(), list)
        assert isinstance(tracker.get_log_pattern(), str)
        
        # Test calculate_progress returns ProgressInfo
        progress = tracker.calculate_progress()
        assert isinstance(progress, ProgressInfo)
        assert hasattr(progress, 'completed_epochs')
        assert hasattr(progress, 'progress_percentage')
        assert hasattr(progress, 'early_stopped')
        assert hasattr(progress, 'early_stopped_at_epoch')
        assert hasattr(progress, 'runner_type')
        assert hasattr(progress, 'total_epochs')


# ============================================================================
# TESTS FOR BaseProgressTracker - EDGE CASES
# ============================================================================

def test_base_progress_tracker_progress_info_dataclass():
    """Test ProgressInfo dataclass behavior."""
    # Test creation with all fields
    progress = ProgressInfo(
        completed_epochs=15,
        progress_percentage=75.0,
        early_stopped=True,
        early_stopped_at_epoch=15,
        runner_type='evaluator',
        total_epochs=20
    )
    
    assert progress.completed_epochs == 15
    assert progress.progress_percentage == 75.0
    assert progress.early_stopped == True
    assert progress.early_stopped_at_epoch == 15
    assert progress.runner_type == 'evaluator'
    assert progress.total_epochs == 20
    
    # Test equality
    progress2 = ProgressInfo(
        completed_epochs=15,
        progress_percentage=75.0,
        early_stopped=True,
        early_stopped_at_epoch=15,
        runner_type='evaluator',
        total_epochs=20
    )
    assert progress == progress2
    
    # Test inequality
    progress3 = ProgressInfo(
        completed_epochs=16,  # Different value
        progress_percentage=75.0,
        early_stopped=True,
        early_stopped_at_epoch=15,
        runner_type='evaluator',
        total_epochs=20
    )
    assert progress != progress3


def test_base_progress_tracker_runner_type_variations(ConcreteProgressTracker):
    """Test tracker with different runner types."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Test trainer type
        trainer_tracker = ConcreteProgressTracker(work_dir, runner_type='trainer')
        trainer_progress = trainer_tracker.get_progress()
        assert trainer_progress.runner_type == 'trainer'
        
        # Test evaluator type
        evaluator_tracker = ConcreteProgressTracker(work_dir, runner_type='evaluator')
        evaluator_progress = evaluator_tracker.get_progress()
        assert evaluator_progress.runner_type == 'evaluator'


def test_base_progress_tracker_deterministic_behavior(ConcreteProgressTracker):
    """Test that tracker behavior is deterministic."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        # Multiple calls should give same result
        results = []
        for _ in range(5):
            progress = tracker.get_progress(force_refresh=True)
            results.append(progress)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
        assert all(r.completed_epochs == 10 for r in results)
        assert all(r.progress_percentage == 50.0 for r in results)


# ============================================================================
# TESTS FOR BaseProgressTracker - INTEGRATION
# ============================================================================

def test_base_progress_tracker_with_various_progress_states(ConcreteProgressTracker):
    """Test tracker with various progress states."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = ConcreteProgressTracker(work_dir)
        
        # Test different progress states
        test_states = [
            # (completed_epochs, progress_percentage, early_stopped, early_stopped_at_epoch)
            (0, 0.0, False, None),      # Just started
            (50, 50.0, False, None),    # Halfway
            (100, 100.0, False, None),  # Complete
            (75, 100.0, True, 75),      # Early stopped
        ]
        
        for completed, percentage, early_stopped, early_stopped_at in test_states:
            test_result = ProgressInfo(
                completed_epochs=completed,
                progress_percentage=percentage,
                early_stopped=early_stopped,
                early_stopped_at_epoch=early_stopped_at,
                runner_type='trainer',
                total_epochs=100
            )
            
            tracker.set_test_progress_result(test_result)
            progress = tracker.get_progress(force_refresh=True)
            
            assert progress.completed_epochs == completed
            assert progress.progress_percentage == percentage
            assert progress.early_stopped == early_stopped
            assert progress.early_stopped_at_epoch == early_stopped_at