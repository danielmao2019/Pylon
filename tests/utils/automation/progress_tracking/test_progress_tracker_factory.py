"""
Test progress_tracker_factory functionality for creating appropriate trackers.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Edge case testing
- Invalid input testing with exception verification
- Determinism testing
"""
import os
import tempfile
import json
import pytest
from utils.automation.progress_tracking.progress_tracker_factory import create_progress_tracker
from utils.automation.progress_tracking.trainer_progress_tracker import TrainerProgressTracker
from utils.automation.progress_tracking.evaluator_progress_tracker import EvaluatorProgressTracker
from utils.automation.progress_tracking.base_progress_tracker import BaseProgressTracker, ProgressInfo
from conftest import create_epoch_files


# ============================================================================
# TESTS FOR create_progress_tracker - BASIC FUNCTIONALITY
# ============================================================================

def test_create_progress_tracker_returns_evaluator_for_evaluation_pattern():
    """Test factory returns EvaluatorProgressTracker for evaluator pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json (evaluator pattern)
        eval_scores = {
            "aggregated": {"accuracy": 0.95},
            "per_datapoint": {"accuracy": [0.9, 0.95, 0.98]}
        }
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = create_progress_tracker(work_dir)
        
        assert isinstance(tracker, EvaluatorProgressTracker)
        assert isinstance(tracker, BaseProgressTracker)  # Should inherit from base
        assert tracker.get_runner_type() == 'evaluator'
        assert tracker.work_dir == work_dir
        assert tracker.config is None


def test_create_progress_tracker_returns_trainer_for_trainer_pattern():
    """Test factory returns TrainerProgressTracker for trainer pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer pattern: epoch_0/validation_scores.json
        create_epoch_files(work_dir, 0)
        
        tracker = create_progress_tracker(work_dir)
        
        assert isinstance(tracker, TrainerProgressTracker)
        assert isinstance(tracker, BaseProgressTracker)  # Should inherit from base
        assert tracker.get_runner_type() == 'trainer'
        assert tracker.work_dir == work_dir
        assert tracker.config is None


def test_create_progress_tracker_with_config():
    """Test factory passes config correctly to created tracker."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluator pattern
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        config = {
            'model': 'test_model',
            'dataset': 'test_dataset',
            'epochs': 100
        }
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, EvaluatorProgressTracker)
        assert tracker.config == config
        assert tracker.work_dir == work_dir


# ============================================================================
# TESTS FOR create_progress_tracker - CONFIG-BASED DETECTION
# ============================================================================

def test_create_progress_tracker_config_evaluator_class():
    """Test factory uses config for evaluator detection when no file patterns."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns
        
        # Mock config with Evaluator class
        config = {
            'runner': {
                'class': type('BaseEvaluator', (), {})  # Mock class with 'Evaluator' in name
            }
        }
        config['runner']['class'].__name__ = 'BaseEvaluator'
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, EvaluatorProgressTracker)
        assert tracker.config == config


def test_create_progress_tracker_config_trainer_class():
    """Test factory uses config for trainer detection when no file patterns."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns
        
        # Mock config with Trainer class
        config = {
            'runner': {
                'class': type('BaseTrainer', (), {})  # Mock class with 'Trainer' in name
            }
        }
        config['runner']['class'].__name__ = 'BaseTrainer'
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, TrainerProgressTracker)
        assert tracker.config == config


def test_create_progress_tracker_config_epochs_field():
    """Test factory detects trainer from 'epochs' field."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns, no runner class
        
        config = {
            'epochs': 100,
            'model': 'test_model'
        }
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, TrainerProgressTracker)
        assert tracker.config == config


# ============================================================================
# TESTS FOR create_progress_tracker - PRECEDENCE RULES
# ============================================================================

def test_create_progress_tracker_evaluator_pattern_takes_precedence():
    """Test that evaluator file pattern takes precedence over trainer pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create both patterns
        create_epoch_files(work_dir, 0)  # Trainer pattern
        
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.8, 0.9, 0.95]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)  # Evaluator pattern
        
        tracker = create_progress_tracker(work_dir)
        
        assert isinstance(tracker, EvaluatorProgressTracker)  # Evaluator wins


def test_create_progress_tracker_file_pattern_over_config():
    """Test that file patterns take precedence over config."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer file pattern
        create_epoch_files(work_dir, 0)
        
        # Config says evaluator
        config = {
            'runner': {
                'class': type('BaseEvaluator', (), {})
            }
        }
        config['runner']['class'].__name__ = 'BaseEvaluator'
        
        tracker = create_progress_tracker(work_dir, config)
        
        # File pattern should win over config
        assert isinstance(tracker, TrainerProgressTracker)


# ============================================================================
# TESTS FOR create_progress_tracker - ERROR HANDLING
# ============================================================================

def test_create_progress_tracker_fail_fast_no_patterns():
    """Test that factory fails fast when no patterns match."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory with some irrelevant files
        with open(os.path.join(work_dir, "some_file.txt"), 'w') as f:
            f.write("irrelevant content")
        
        with pytest.raises(ValueError) as exc_info:
            create_progress_tracker(work_dir)
        
        error_msg = str(exc_info.value)
        assert "Unable to detect runner type" in error_msg
        assert work_dir in error_msg
        assert "Available files:" in error_msg
        assert "some_file.txt" in error_msg


def test_create_progress_tracker_fail_fast_nonexistent_directory():
    """Test fail fast behavior with nonexistent directory."""
    nonexistent_dir = "/this/path/does/not/exist"
    
    with pytest.raises(ValueError) as exc_info:
        create_progress_tracker(nonexistent_dir)
    
    error_msg = str(exc_info.value)
    assert "Unable to detect runner type" in error_msg
    assert nonexistent_dir in error_msg


def test_create_progress_tracker_invalid_config():
    """Test factory with invalid config that doesn't help detection."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Config with runner class that doesn't contain Trainer or Evaluator
        config = {
            'runner': {
                'class': type('SomeOtherClass', (), {})
            }
        }
        config['runner']['class'].__name__ = 'SomeOtherClass'
        
        with pytest.raises(ValueError) as exc_info:
            create_progress_tracker(work_dir, config)
        
        assert "Unable to detect runner type" in str(exc_info.value)


# ============================================================================
# TESTS FOR create_progress_tracker - EDGE CASES
# ============================================================================

def test_create_progress_tracker_various_config_formats():
    """Test factory with various config formats."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluator pattern
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Test various config formats
        test_configs = [
            None,                                    # No config
            {},                                      # Empty config
            {'model': 'test'},                      # Simple config
            {'runner': {}},                         # Empty runner
            {'runner': {'class': None}},            # None runner class
        ]
        
        for config in test_configs:
            tracker = create_progress_tracker(work_dir, config)
            assert isinstance(tracker, EvaluatorProgressTracker)
            assert tracker.config == config


def test_create_progress_tracker_config_string_class_name():
    """Test factory when runner class is a string instead of actual class."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, rely on config
        
        config = {
            'runner': {
                'class': 'BaseEvaluator'  # String instead of class
            }
        }
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, EvaluatorProgressTracker)
        assert tracker.config == config


# ============================================================================
# TESTS FOR create_progress_tracker - DETERMINISM
# ============================================================================

def test_create_progress_tracker_deterministic():
    """Test that factory creates consistent tracker types."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluator pattern
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Create multiple trackers
        trackers = [create_progress_tracker(work_dir) for _ in range(5)]
        
        # All should be the same type
        assert all(isinstance(t, EvaluatorProgressTracker) for t in trackers)
        assert all(t.work_dir == work_dir for t in trackers)
        assert all(t.config is None for t in trackers)


# ============================================================================
# TESTS FOR create_progress_tracker - INTEGRATION
# ============================================================================

def test_create_progress_tracker_integration_with_real_scenarios():
    """Test factory with realistic trainer and evaluator scenarios."""
    
    # Test realistic trainer scenario
    with tempfile.TemporaryDirectory() as work_dir:
        # Create multiple epoch files (realistic trainer)
        for epoch_idx in range(5):
            create_epoch_files(work_dir, epoch_idx)
        
        config = {
            'epochs': 100,
            'model': 'resnet50',
            'optimizer': 'adam',
            'dataset': 'imagenet'
        }
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, TrainerProgressTracker)
        assert tracker.config == config
        
        # Verify the tracker was created correctly (don't call get_progress due to config file dependencies)
        assert tracker.get_runner_type() == 'trainer'
        assert tracker.get_expected_files() == ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]
    
    # Test realistic evaluator scenario
    with tempfile.TemporaryDirectory() as work_dir:
        # Create realistic evaluation results
        eval_scores = {
            "aggregated": {
                "accuracy": 0.9234,
                "precision": 0.8912,
                "recall": 0.9156,
                "f1_score": 0.9032
            },
            "per_datapoint": {
                "accuracy": [0.92, 0.91, 0.94, 0.93],
                "precision": [0.89, 0.88, 0.91, 0.90],
                "recall": [0.91, 0.92, 0.93, 0.90],
                "f1_score": [0.90, 0.90, 0.92, 0.90]
            }
        }
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        config = {
            'model': 'bert_large',
            'dataset': 'test_set',
            'batch_size': 32
        }
        
        tracker = create_progress_tracker(work_dir, config)
        
        assert isinstance(tracker, EvaluatorProgressTracker)
        assert tracker.config == config
        
        # Should be able to get progress
        progress = tracker.get_progress()
        assert progress.runner_type == 'evaluator'
        assert progress.completed_epochs == 1
        assert progress.progress_percentage == 100.0


def test_create_progress_tracker_validates_created_instances():
    """Test that factory creates properly initialized tracker instances."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer pattern
        create_epoch_files(work_dir, 0)
        
        config = {'epochs': 50, 'model': 'test_model'}
        
        tracker = create_progress_tracker(work_dir, config)
        
        # Verify the created instance is properly initialized
        assert isinstance(tracker, TrainerProgressTracker)
        assert isinstance(tracker, BaseProgressTracker)
        
        # Verify it has all required attributes
        assert hasattr(tracker, 'work_dir')
        assert hasattr(tracker, 'config')
        assert hasattr(tracker, '_cache')
        assert hasattr(tracker, '_cache_time')
        
        # Verify abstract methods are implemented
        assert callable(tracker.get_runner_type)
        assert callable(tracker.get_expected_files)
        assert callable(tracker.get_log_pattern)
        assert callable(tracker.calculate_progress)
        assert callable(tracker.get_progress)
        
        # Verify it can actually work
        progress = tracker.get_progress()
        assert isinstance(progress, ProgressInfo)
        assert progress.runner_type == 'trainer'
