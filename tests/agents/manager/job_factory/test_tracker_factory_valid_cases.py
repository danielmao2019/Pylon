"""
Test tracker_factory functionality for creating appropriate trackers - VALID CASES.

Following CLAUDE.md testing patterns:
- Correctness verification with known inputs/outputs  
- Determinism testing
- Integration testing
"""
import os
import tempfile
import json
from agents.manager.manager import Manager


# ============================================================================
# TESTS FOR create_tracker - BASIC FUNCTIONALITY
# ============================================================================

def test_detect_runner_evaluator_for_evaluation_pattern():
    """Test factory returns EvaluatorTracker for evaluator pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json (evaluator pattern)
        eval_scores = {
            "aggregated": {"accuracy": 0.95},
            "per_datapoint": {"accuracy": [0.9, 0.95, 0.98]}
        }
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'evaluator'


def test_detect_runner_trainer_for_trainer_pattern(create_epoch_files):
    """Test factory returns TrainerTracker for trainer pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer pattern: epoch_0/validation_scores.json
        create_epoch_files(work_dir, 0)
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'trainer'


def test_detect_runner_with_config():
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
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, config) == 'evaluator'


# ============================================================================
# TESTS FOR create_tracker - CONFIG-BASED DETECTION
# ============================================================================

def test_detect_runner_config_evaluator_class():
    """Test factory uses config for evaluator detection when no file patterns."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns
        
        # Mock config with Evaluator class (direct class reference)
        mock_evaluator_class = type('BaseEvaluator', (), {})
        config = {
            'runner': mock_evaluator_class
        }
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, config) == 'evaluator'


def test_detect_runner_config_trainer_class():
    """Test factory uses config for trainer detection when no file patterns."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns
        
        # Mock config with Trainer class (direct class reference)
        mock_trainer_class = type('BaseTrainer', (), {})
        config = {
            'runner': mock_trainer_class
        }
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, config) == 'trainer'


# ============================================================================
# TESTS FOR create_tracker - PRECEDENCE RULES
# ============================================================================

def test_detect_runner_evaluator_pattern_takes_precedence(create_epoch_files):
    """Test that evaluator file pattern takes precedence over trainer pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create both patterns
        create_epoch_files(work_dir, 0)  # Trainer pattern
        
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.8, 0.9, 0.95]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)  # Evaluator pattern
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, None) == 'evaluator'


def test_detect_runner_file_pattern_over_config(create_epoch_files):
    """Test that file patterns take precedence over config."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer file pattern
        create_epoch_files(work_dir, 0)
        
        # Config says evaluator (direct class reference)
        mock_evaluator_class = type('BaseEvaluator', (), {})
        config = {
            'runner': mock_evaluator_class
        }
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        assert m._detect_runner_type(work_dir, config) == 'trainer'


# ============================================================================
# TESTS FOR create_tracker - EDGE CASES
# ============================================================================

def test_detect_runner_various_config_formats():
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
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        for config in test_configs:
            assert m._detect_runner_type(work_dir, config) == 'evaluator'


# ============================================================================
# TESTS FOR create_tracker - DETERMINISM
# ============================================================================

def test_detect_runner_deterministic():
    """Test that factory creates consistent tracker types."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluator pattern
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Create multiple trackers
        m = Manager(config_files=[], epochs=1, system_monitors={})
        results = [m._detect_runner_type(work_dir, None) for _ in range(5)]
        assert all(r == 'evaluator' for r in results)


# ============================================================================
# TESTS FOR create_tracker - INTEGRATION
# ============================================================================

def test_create_tracker_integration_with_real_scenarios(create_epoch_files):
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
        
        tracker = create_tracker(work_dir, config)
        
        assert isinstance(tracker, TrainerTracker)
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
        
        tracker = create_tracker(work_dir, config)
        
        assert isinstance(tracker, EvaluatorTracker)
        assert tracker.config == config
        
        # Should be able to get progress
        progress = tracker.get_progress()
        assert progress.runner_type == 'evaluator'
        assert progress.completed_epochs == 1
        assert progress.progress_percentage == 100.0


def test_create_tracker_validates_created_instances(create_epoch_files):
    """Test that factory creates properly initialized tracker instances."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer pattern
        create_epoch_files(work_dir, 0)
        
        config = {'epochs': 50, 'model': 'test_model'}
        
        tracker = create_tracker(work_dir, config)
        
        # Verify the created instance is properly initialized
        assert isinstance(tracker, TrainerTracker)
        assert isinstance(tracker, BaseTracker)
        
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
        
        # Verify the tracker is properly initialized without calling get_progress()
        # (get_progress() requires complex config file setup which is not the focus of this test)
        assert tracker.get_runner_type() == 'trainer'
        assert len(tracker.get_expected_files()) > 0
