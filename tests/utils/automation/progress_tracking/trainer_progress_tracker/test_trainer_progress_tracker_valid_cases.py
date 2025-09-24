"""
Test TrainerProgressTracker functionality for trainer progress tracking - VALID CASES.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Determinism testing
- Integration testing
"""
import os
import tempfile
import json
import pytest
from agents.tracker import ProgressInfo
from agents.tracker.trainer_progress_tracker import TrainerProgressTracker


# ============================================================================
# TESTS FOR TrainerProgressTracker - BASIC FUNCTIONALITY
# ============================================================================

def test_trainer_progress_tracker_initialization():
    """Test that TrainerProgressTracker initializes correctly."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'some_config': 'value'}
        tracker = TrainerProgressTracker(work_dir, config)
        
        assert tracker.work_dir == work_dir
        assert tracker.config == config
        assert tracker.get_runner_type() == 'trainer'
        assert tracker.get_expected_files() == ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
        assert tracker.get_log_pattern() == "train_val*.log"


def test_trainer_progress_tracker_initialization_no_config():
    """Test initialization without config."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = TrainerProgressTracker(work_dir)
        
        assert tracker.work_dir == work_dir
        assert tracker.config is None
        assert tracker.get_runner_type() == 'trainer'


# ============================================================================
# TESTS FOR TrainerProgressTracker - PROGRESS CALCULATION (FAST PATH)
# ============================================================================

def test_trainer_progress_tracker_fast_path_normal_run(create_progress_json):
    """Test progress calculation using existing progress.json (fast path)."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'model': 'some_model'}
        
        # Create existing progress.json for normal run (57/100 epochs)
        create_progress_json(work_dir, completed_epochs=57, early_stopped=False, tot_epochs=100)
        
        tracker = TrainerProgressTracker(work_dir, config)
        progress = tracker.get_progress()
        
        assert isinstance(progress, ProgressInfo)
        assert progress.runner_type == 'trainer'
        assert progress.completed_epochs == 57
        assert abs(progress.progress_percentage - 57.0) < 0.01  # Handle floating point precision
        assert progress.total_epochs == 100  # From config
        assert progress.early_stopped == False
        assert progress.early_stopped_at_epoch is None


def test_trainer_progress_tracker_fast_path_early_stopped(create_progress_json):
    """Test progress calculation with early stopping (fast path)."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'epochs': 100, 'model': 'some_model'}
        
        # Create existing progress.json for early stopped run
        create_progress_json(work_dir, completed_epochs=57, early_stopped=True, 
                           early_stopped_at_epoch=57, tot_epochs=100)
        
        tracker = TrainerProgressTracker(work_dir, config)
        progress = tracker.get_progress()
        
        assert isinstance(progress, ProgressInfo)
        assert progress.runner_type == 'trainer'
        assert progress.completed_epochs == 57
        assert progress.progress_percentage == 100.0  # Early stopped shows 100%
        assert progress.total_epochs == 100  # From config
        assert progress.early_stopped == True
        assert progress.early_stopped_at_epoch == 57


def test_trainer_progress_tracker_no_config_epochs(create_progress_json):
    """Test progress calculation when config doesn't have epochs."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'model': 'some_model'}  # No epochs field
        
        # Create existing progress.json
        create_progress_json(work_dir, completed_epochs=30, early_stopped=False, tot_epochs=100)
        
        tracker = TrainerProgressTracker(work_dir, config)
        progress = tracker.get_progress()
        
        assert isinstance(progress, ProgressInfo)
        assert progress.runner_type == 'trainer'
        assert progress.completed_epochs == 30
        assert progress.total_epochs is None  # No epochs in config


# ============================================================================
# TESTS FOR TrainerProgressTracker - PROGRESS CALCULATION (SLOW PATH)
# ============================================================================

@pytest.mark.parametrize("completed_epochs,expected_completed", [
    (0, 0),      # No epochs completed
    (1, 1),      # One epoch completed  
    (25, 25),    # Quarter completed
    (50, 50),    # Half completed
    (75, 75),    # Three quarters completed
    (99, 99),    # Almost complete
])
def test_trainer_progress_tracker_slow_path_normal_runs(completed_epochs, expected_completed, create_epoch_files, create_real_config):
    """Test slow path (no progress.json) for various completion levels."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_trainer_run")
        config_path = os.path.join(configs_dir, "test_trainer_run.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create epoch files for completed epochs
        for epoch_idx in range(completed_epochs):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config (no early stopping)
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            config = {'epochs': 100, 'model': 'test_model'}
            tracker = TrainerProgressTracker(work_dir, config)
            progress = tracker.get_progress()
            
            # Should return ProgressInfo dataclass
            assert isinstance(progress, ProgressInfo), f"Expected ProgressInfo, got {type(progress)}"
            assert progress.runner_type == 'trainer'
            assert progress.completed_epochs == expected_completed
            assert progress.total_epochs == 100  # From config
            assert progress.early_stopped == False
            assert progress.early_stopped_at_epoch is None
            
            # Verify progress.json was created
            progress_file = os.path.join(work_dir, "progress.json")
            assert os.path.exists(progress_file), "progress.json should have been created"
            
        finally:
            os.chdir(original_cwd)


def test_trainer_progress_tracker_slow_path_with_early_stopping(create_epoch_files, create_real_config):
    """Test slow path calculation when early stopping is detected."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_early_stop")
        config_path = os.path.join(configs_dir, "test_early_stop.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create several epoch files with early stopping pattern
        # (this is detected by session_progress logic)
        for epoch_idx in range(30):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config with early stopping
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=True, patience=5)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            config = {'epochs': 100, 'model': 'test_model'}
            tracker = TrainerProgressTracker(work_dir, config)
            progress = tracker.get_progress()
            
            # Should detect the epochs correctly
            assert isinstance(progress, ProgressInfo)
            assert progress.runner_type == 'trainer'
            assert progress.completed_epochs == 30
            assert progress.total_epochs == 100
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR TrainerProgressTracker - CACHING
# ============================================================================

def test_trainer_progress_tracker_caching(create_progress_json, create_epoch_files, create_real_config):
    """Test that progress caching works correctly."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_caching")
        config_path = os.path.join(configs_dir, "test_caching.py")
        
        os.makedirs(work_dir, exist_ok=True)
        config = {'epochs': 100, 'model': 'test_model'}
        
        # Create actual epoch files (42 epochs)
        for epoch_idx in range(42):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=42, early_stopped=False, tot_epochs=100)
        
        # Create real config (needed for force_progress_recompute)
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            tracker = TrainerProgressTracker(work_dir, config)
            
            # First call should compute progress
            progress1 = tracker.get_progress()
            assert progress1.completed_epochs == 42
            
            # Second call should use cache (within cache timeout)
            progress2 = tracker.get_progress()
            assert progress2 == progress1
            
            # Force progress recompute should bypass cache
            progress3 = tracker.get_progress(force_progress_recompute=True)
            assert progress3.completed_epochs == 42  # Same result but forced recalculation
            
        finally:
            os.chdir(original_cwd)


def test_trainer_progress_tracker_progress_json_creation(create_epoch_files, create_real_config):
    """Test that progress.json is created with correct format."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_progress_creation")
        config_path = os.path.join(configs_dir, "test_progress_creation.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create some epoch files
        for epoch_idx in range(10):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            config = {'epochs': 100, 'model': 'test_model'}
            tracker = TrainerProgressTracker(work_dir, config)
            progress = tracker.get_progress()
            
            # Check that progress.json was created
            progress_file = os.path.join(work_dir, "progress.json")
            assert os.path.exists(progress_file)
            
            # Verify content
            with open(progress_file, 'r') as f:
                saved_progress = json.load(f)
            
            assert saved_progress['runner_type'] == 'trainer'
            assert saved_progress['completed_epochs'] == 10
            assert saved_progress['early_stopped'] == False
            assert saved_progress['early_stopped_at_epoch'] is None
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR TrainerProgressTracker - DETERMINISM
# ============================================================================

def test_trainer_progress_tracker_deterministic(create_progress_json, create_epoch_files, create_real_config):
    """Test that progress calculation is deterministic across multiple calls."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_deterministic")
        config_path = os.path.join(configs_dir, "test_deterministic.py")
        
        os.makedirs(work_dir, exist_ok=True)
        config = {'epochs': 100, 'model': 'test_model'}
        
        # Create actual epoch files (75 epochs)
        for epoch_idx in range(75):
            create_epoch_files(work_dir, epoch_idx)
        
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=75, early_stopped=False, tot_epochs=100)
        
        # Create real config (needed for force_progress_recompute)
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            tracker = TrainerProgressTracker(work_dir, config)
            
            # Multiple calls should give same result
            results = []
            for _ in range(5):
                progress = tracker.get_progress(force_progress_recompute=True)
                results.append(progress)
            
            # All results should be identical
            assert all(r == results[0] for r in results)
            assert all(r.completed_epochs == 75 for r in results)
            assert all(r.progress_percentage == 75.0 for r in results)
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR TrainerProgressTracker - EDGE CASES (VALID ONLY)
# ============================================================================

def test_trainer_progress_tracker_empty_work_dir(create_real_config):
    """Test progress calculation with empty work directory."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_empty_trainer")
        config_path = os.path.join(configs_dir, "test_empty_trainer.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            config = {'epochs': 100, 'model': 'test_model'}
            tracker = TrainerProgressTracker(work_dir, config)
            progress = tracker.get_progress()
            
            # Should return ProgressInfo dataclass with 0 completed epochs
            assert isinstance(progress, ProgressInfo)
            assert progress.runner_type == 'trainer'
            assert progress.completed_epochs == 0
            assert progress.total_epochs == 100
            assert progress.early_stopped == False
            assert progress.early_stopped_at_epoch is None
            
        finally:
            os.chdir(original_cwd)


# ============================================================================
# TESTS FOR TrainerProgressTracker - INTEGRATION
# ============================================================================

def test_trainer_progress_tracker_with_various_configs(create_progress_json):
    """Test trainer tracker with various config values."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create progress.json
        create_progress_json(work_dir, completed_epochs=50, early_stopped=False, tot_epochs=200)
        
        # Test with various config formats
        test_configs = [
            {'epochs': 200, 'model': 'resnet'},
            {'epochs': 200, 'model': 'vit', 'optimizer': 'adam'},
            {'epochs': 200},  # Minimal config
        ]
        
        for config in test_configs:
            tracker = TrainerProgressTracker(work_dir, config)
            progress = tracker.get_progress()
            
            # Config should affect total_epochs
            assert progress.total_epochs == 200
            assert progress.completed_epochs == 50
            assert progress.runner_type == 'trainer'


def test_trainer_progress_tracker_integration_with_session_progress(create_epoch_files, create_real_config):
    """Test integration with moved session_progress module."""
    with tempfile.TemporaryDirectory() as temp_root:
        # Create directory structure that matches cfg_log_conversion pattern
        logs_dir = os.path.join(temp_root, "logs")
        configs_dir = os.path.join(temp_root, "configs")
        work_dir = os.path.join(logs_dir, "test_integration")
        config_path = os.path.join(configs_dir, "test_integration.py")
        
        os.makedirs(work_dir, exist_ok=True)
        
        # Create realistic training progress (15 epochs completed)
        for epoch_idx in range(15):
            create_epoch_files(work_dir, epoch_idx, validation_score=0.5 - epoch_idx * 0.01)
        
        # Create real config
        create_real_config(config_path, work_dir, epochs=100, early_stopping_enabled=False)
        
        # Change to temp_root so relative paths work
        original_cwd = os.getcwd()
        os.chdir(temp_root)
        
        try:
            config = {'epochs': 100, 'model': 'integration_test'}
            tracker = TrainerProgressTracker(work_dir, config)
            progress = tracker.get_progress()
            
            # Should correctly integrate with session_progress logic
            assert isinstance(progress, ProgressInfo)
            assert progress.runner_type == 'trainer'
            assert progress.completed_epochs == 15
            assert progress.total_epochs == 100
            assert abs(progress.progress_percentage - 15.0) < 0.01
            assert progress.early_stopped == False
            
            # Verify the underlying session_progress logic worked
            progress_file = os.path.join(work_dir, "progress.json")
            assert os.path.exists(progress_file)
            
        finally:
            os.chdir(original_cwd)
