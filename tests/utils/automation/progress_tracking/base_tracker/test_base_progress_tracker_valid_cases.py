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
from agents.tracker.base_progress_tracker import BaseProgressTracker, ProgressInfo


# ============================================================================
# TESTS FOR BaseProgressTracker - INITIALIZATION
# ============================================================================

def test_base_progress_tracker_trainer_initialization_with_config(TrainerProgressTrackerForTesting):
    """Test BaseProgressTracker initialization with config for TrainerProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'model': 'test_model'}
        tracker = TrainerProgressTrackerForTesting(work_dir, config)
        
        assert tracker.work_dir == work_dir
        assert tracker.config == config
        assert tracker._cache is None
        assert tracker._cache_time is None


def test_base_progress_tracker_evaluator_initialization_with_config(EvaluatorProgressTrackerForTesting):
    """Test BaseProgressTracker initialization with config for EvaluatorProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'model': 'test_model'}
        tracker = EvaluatorProgressTrackerForTesting(work_dir, config)
        
        assert tracker.work_dir == work_dir
        assert tracker.config == config
        assert tracker._cache is None
        assert tracker._cache_time is None


def test_base_progress_tracker_trainer_initialization_without_config(TrainerProgressTrackerForTesting):
    """Test BaseProgressTracker initialization without config for TrainerProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = TrainerProgressTrackerForTesting(work_dir)
        
        assert tracker.work_dir == work_dir
        assert tracker.config is None
        assert tracker._cache is None
        assert tracker._cache_time is None


def test_base_progress_tracker_evaluator_initialization_without_config(EvaluatorProgressTrackerForTesting):
    """Test BaseProgressTracker initialization without config for EvaluatorProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        
        assert tracker.work_dir == work_dir
        assert tracker.config is None
        assert tracker._cache is None
        assert tracker._cache_time is None


# ============================================================================
# TESTS FOR BaseProgressTracker - CACHING MECHANISM
# ============================================================================

def test_base_progress_tracker_trainer_caching_basic(TrainerProgressTrackerForTesting, create_progress_json):
    """Test basic caching functionality for TrainerProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create progress.json for fast path
        create_progress_json(work_dir, completed_epochs=10, early_stopped=False, tot_epochs=100)
        
        tracker = TrainerProgressTrackerForTesting(work_dir)
        
        # First call should calculate and cache
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 10
        assert tracker._cache is not None
        assert tracker._cache_time is not None
        
        # Second call should return cached result
        progress2 = tracker.get_progress()
        assert progress2 == progress1


def test_base_progress_tracker_evaluator_caching_basic(EvaluatorProgressTrackerForTesting):
    """Test basic caching functionality for EvaluatorProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json for completed evaluation
        import json
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        
        # First call should calculate and cache
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 1  # Binary for evaluators
        assert tracker._cache is not None
        assert tracker._cache_time is not None
        
        # Second call should return cached result
        progress2 = tracker.get_progress()
        assert progress2 == progress1


def test_base_progress_tracker_trainer_force_progress_recompute(TrainerProgressTrackerForTesting, create_progress_json, create_epoch_files, create_real_config):
    """Test force progress recompute bypasses cache for TrainerProgressTracker."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_force_recompute")
        config_path = os.path.join(configs_dir, "test_force_recompute.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create outdated progress.json showing 2 epochs
        create_progress_json(work_dir, completed_epochs=2, early_stopped=False, tot_epochs=100)
        
        tracker = TrainerProgressTrackerForTesting(work_dir)
        
        # First call - should return cached result
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 2
        
        # Now create actual epoch files (5 epochs) 
        for epoch_idx in range(5):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config for slow path
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            # Normal call should still return cached result
            progress2 = tracker.get_progress()
            assert progress2.completed_epochs == 2  # Still cached
            
            # Force progress recompute should get new result from filesystem
            progress3 = tracker.get_progress(force_progress_recompute=True)
            assert progress3.completed_epochs == 5  # New result from filesystem
            
        finally:
            os.chdir(original_cwd)


def test_base_progress_tracker_evaluator_force_progress_recompute(EvaluatorProgressTrackerForTesting):
    """Test force progress recompute bypasses cache for EvaluatorProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Start with no evaluation file (incomplete)
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        
        # First call - should show incomplete
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 0
        
        # Create evaluation_scores.json (complete evaluation)
        import json
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Normal call should still return cached result
        progress2 = tracker.get_progress()
        assert progress2.completed_epochs == 0  # Still cached
        
        # Force progress recompute should get new result
        progress3 = tracker.get_progress(force_progress_recompute=True)
        assert progress3.completed_epochs == 1  # New result - now complete


def test_base_progress_tracker_trainer_cache_timeout(TrainerProgressTrackerForTesting, create_progress_json):
    """Test that cache expires after timeout for TrainerProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create initial progress.json
        create_progress_json(work_dir, completed_epochs=10, early_stopped=False, tot_epochs=100)
        
        # Use very short cache timeout for testing
        tracker = TrainerProgressTrackerForTesting(work_dir)
        tracker._cache_timeout = 0.1  # 100ms
        
        # First call
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 10
        
        # Update the progress.json file to simulate new progress
        create_progress_json(work_dir, completed_epochs=30, early_stopped=False, tot_epochs=100)
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Should get new result due to cache timeout
        progress2 = tracker.get_progress()
        assert progress2.completed_epochs == 30


def test_base_progress_tracker_evaluator_cache_timeout(EvaluatorProgressTrackerForTesting):
    """Test that cache expires after timeout for EvaluatorProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Use very short cache timeout for testing
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        tracker._cache_timeout = 0.1  # 100ms
        
        # First call - no evaluation file (incomplete)
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 0
        
        # Create evaluation_scores.json (complete evaluation)
        import json
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Should get new result due to cache timeout
        progress2 = tracker.get_progress()
        assert progress2.completed_epochs == 1


# ============================================================================
# TESTS FOR BaseProgressTracker - PROGRESS.JSON HANDLING
# ============================================================================

def test_base_progress_tracker_trainer_saves_progress_json(TrainerProgressTrackerForTesting, create_epoch_files, create_real_config):
    """Test that get_progress saves progress.json file for TrainerProgressTracker."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_saves_progress")
        config_path = os.path.join(configs_dir, "test_saves_progress.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create some epoch files for realistic test
        for epoch_idx in range(5):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            tracker = TrainerProgressTrackerForTesting(work_dir)
            
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
            
        finally:
            os.chdir(original_cwd)


def test_base_progress_tracker_evaluator_saves_progress_json(EvaluatorProgressTrackerForTesting):
    """Test that get_progress saves progress.json file for EvaluatorProgressTracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json for completed evaluation
        import json
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        
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


# Note: JSON save error testing moved to individual tracker tests where it's more realistic


# ============================================================================
# TESTS FOR BaseProgressTracker - ABSTRACT METHOD CONTRACTS
# ============================================================================

def test_base_progress_tracker_trainer_implementation_contracts(TrainerProgressTrackerForTesting, create_progress_json):
    """Test that TrainerProgressTracker follows expected contracts."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create progress.json for realistic test
        create_progress_json(work_dir, completed_epochs=15, early_stopped=False, tot_epochs=100)
        
        tracker = TrainerProgressTrackerForTesting(work_dir)
        
        # Test abstract method implementations
        assert tracker.get_runner_type() == 'trainer'
        assert isinstance(tracker.get_expected_files(), list)
        assert len(tracker.get_expected_files()) > 0
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
        assert progress.runner_type == 'trainer'


def test_base_progress_tracker_evaluator_implementation_contracts(EvaluatorProgressTrackerForTesting):
    """Test that EvaluatorProgressTracker follows expected contracts."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json for realistic test
        import json
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        
        # Test abstract method implementations
        assert tracker.get_runner_type() == 'evaluator'
        assert isinstance(tracker.get_expected_files(), list)
        assert len(tracker.get_expected_files()) > 0
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
        assert progress.runner_type == 'evaluator'


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


# Note: Runner type variations are tested via separate TrainerProgressTracker and EvaluatorProgressTracker tests


def test_base_progress_tracker_trainer_deterministic_behavior(TrainerProgressTrackerForTesting, create_epoch_files, create_real_config):
    """Test that TrainerProgressTracker behavior is deterministic."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_deterministic")
        config_path = os.path.join(configs_dir, "test_deterministic.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create epoch files for consistent results
        for epoch_idx in range(10):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            tracker = TrainerProgressTrackerForTesting(work_dir)
            
            # Multiple calls should give same result
            results = []
            for _ in range(5):
                progress = tracker.get_progress(force_progress_recompute=True)
                results.append(progress)
            
            # All results should be identical
            assert all(r == results[0] for r in results)
            assert all(r.completed_epochs == 10 for r in results)
            assert all(r.progress_percentage == 10.0 for r in results)
            
        finally:
            os.chdir(original_cwd)


def test_base_progress_tracker_evaluator_deterministic_behavior(EvaluatorProgressTrackerForTesting):
    """Test that EvaluatorProgressTracker behavior is deterministic."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json for consistent results
        import json
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorProgressTrackerForTesting(work_dir)
        
        # Multiple calls should give same result
        results = []
        for _ in range(5):
            progress = tracker.get_progress(force_progress_recompute=True)
            results.append(progress)
        
        # All results should be identical
        assert all(r == results[0] for r in results)
        assert all(r.completed_epochs == 1 for r in results)
        assert all(r.progress_percentage == 100.0 for r in results)


# ============================================================================
# TESTS FOR BaseProgressTracker - INTEGRATION
# ============================================================================

# Note: Various progress states are tested via individual TrainerProgressTracker and EvaluatorProgressTracker tests
# with realistic file structures representing different completion levels
