"""
Test EvaluatorTracker functionality for evaluator progress tracking - VALID CASES.

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
from agents.tracker.evaluator_tracker import EvaluatorTracker


# ============================================================================
# TESTS FOR EvaluatorTracker - BASIC FUNCTIONALITY
# ============================================================================

def test_evaluator_tracker_initialization():
    """Test that EvaluatorTracker initializes correctly."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {'some_config': 'value'}
        tracker = EvaluatorTracker(work_dir, config)
        
        assert tracker.work_dir == work_dir
        assert tracker.config == config
        assert tracker.get_runner_type() == 'evaluator'
        assert tracker.get_expected_files() == ["evaluation_scores.json"]
        assert tracker.get_log_pattern() == "eval_*.log"


def test_evaluator_tracker_initialization_no_config():
    """Test initialization without config."""
    with tempfile.TemporaryDirectory() as work_dir:
        tracker = EvaluatorTracker(work_dir)
        
        assert tracker.work_dir == work_dir
        assert tracker.config is None
        assert tracker.get_runner_type() == 'evaluator'


# ============================================================================
# TESTS FOR EvaluatorTracker - PROGRESS CALCULATION
# ============================================================================

def test_evaluator_tracker_complete_evaluation():
    """Test progress calculation when evaluation is complete."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json (complete evaluation)
        eval_scores = {
            "aggregated": {"accuracy": 0.95, "loss": 0.05},
            "per_datapoint": {
                "accuracy": [0.9, 0.95, 0.98],
                "loss": [0.1, 0.05, 0.02]
            }
        }
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorTracker(work_dir)
        progress = tracker.get_progress()
        
        assert isinstance(progress, ProgressInfo)
        assert progress.runner_type == 'evaluator'
        assert progress.completed_epochs == 1  # Binary: 1 if complete
        assert progress.progress_percentage == 100.0  # Binary: 100% if complete
        assert progress.total_epochs == 1  # Conceptually 1 epoch for evaluators
        assert progress.early_stopped == False  # Not applicable for evaluators
        assert progress.early_stopped_at_epoch is None


def test_evaluator_tracker_incomplete_evaluation():
    """Test progress calculation when evaluation is not complete."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty work directory (no evaluation_scores.json)
        
        tracker = EvaluatorTracker(work_dir)
        progress = tracker.get_progress()
        
        assert isinstance(progress, ProgressInfo)
        assert progress.runner_type == 'evaluator'
        assert progress.completed_epochs == 0  # Binary: 0 if not complete
        assert progress.progress_percentage == 0.0  # Binary: 0% if not complete
        assert progress.total_epochs == 1  # Conceptually 1 epoch for evaluators
        assert progress.early_stopped == False  # Not applicable for evaluators
        assert progress.early_stopped_at_epoch is None


def test_evaluator_tracker_empty_evaluation_file():
    """Test progress calculation with empty evaluation file."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create empty evaluation_scores.json
        eval_file = os.path.join(work_dir, "evaluation_scores.json")
        with open(eval_file, 'w') as f:
            f.write("")  # Empty file
        
        tracker = EvaluatorTracker(work_dir)
        progress = tracker.get_progress()
        
        # Empty file should be treated as incomplete
        assert progress.completed_epochs == 0
        assert progress.progress_percentage == 0.0


def test_evaluator_tracker_malformed_evaluation_file():
    """Test that malformed evaluation file is treated as incomplete."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create malformed evaluation_scores.json
        eval_file = os.path.join(work_dir, "evaluation_scores.json")
        with open(eval_file, 'w') as f:
            f.write("invalid json content {")  # Malformed JSON
        
        tracker = EvaluatorTracker(work_dir)
        # Should not crash, should treat as incomplete
        progress = tracker.get_progress()
        
        assert progress.completed_epochs == 0
        assert progress.progress_percentage == 0.0


# ============================================================================
# TESTS FOR EvaluatorTracker - FILE CHECKING
# ============================================================================

def test_evaluator_tracker_check_files_exist_valid():
    """Test _check_files_exist with valid evaluation file."""
    with tempfile.TemporaryDirectory() as work_dir:
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorTracker(work_dir)
        assert tracker._check_files_exist() == True


def test_evaluator_tracker_check_files_exist_missing():
    """Test _check_files_exist with missing evaluation file."""
    with tempfile.TemporaryDirectory() as work_dir:
        # No evaluation_scores.json file
        
        tracker = EvaluatorTracker(work_dir)
        assert tracker._check_files_exist() == False


def test_evaluator_tracker_check_files_exist_empty():
    """Test _check_files_exist with empty evaluation file."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create empty file
        eval_file = os.path.join(work_dir, "evaluation_scores.json")
        with open(eval_file, 'w') as f:
            pass  # Empty file
        
        tracker = EvaluatorTracker(work_dir)
        assert tracker._check_files_exist() == False  # Empty file should fail


# ============================================================================
# TESTS FOR EvaluatorTracker - CACHING
# ============================================================================

def test_evaluator_tracker_caching():
    """Test that progress caching works correctly."""
    with tempfile.TemporaryDirectory() as work_dir:
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorTracker(work_dir)
        
        # First call should compute progress
        progress1 = tracker.get_progress()
        assert progress1.completed_epochs == 1
        
        # Second call should use cache (within cache timeout)
        progress2 = tracker.get_progress()
        assert progress2 == progress1
        
        # Force progress recompute should bypass cache
        progress3 = tracker.get_progress(force_progress_recompute=True)
        assert progress3.completed_epochs == 1  # Same result but forced recalculation


def test_evaluator_tracker_progress_json_creation():
    """Test that progress.json is created with correct format."""
    with tempfile.TemporaryDirectory() as work_dir:
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorTracker(work_dir)
        progress = tracker.get_progress()
        
        # Check that progress.json was created
        progress_file = os.path.join(work_dir, "progress.json")
        assert os.path.exists(progress_file)
        
        # Verify content
        with open(progress_file, 'r') as f:
            saved_progress = json.load(f)
        
        assert saved_progress['runner_type'] == 'evaluator'
        assert saved_progress['completed_epochs'] == 1
        assert saved_progress['progress_percentage'] == 100.0
        assert saved_progress['total_epochs'] == 1
        assert saved_progress['early_stopped'] == False
        assert saved_progress['early_stopped_at_epoch'] is None


# ============================================================================
# TESTS FOR EvaluatorTracker - DETERMINISM
# ============================================================================

def test_evaluator_tracker_deterministic():
    """Test that progress calculation is deterministic across multiple calls."""
    with tempfile.TemporaryDirectory() as work_dir:
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorTracker(work_dir)
        
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
# TESTS FOR EvaluatorTracker - EDGE CASES
# ============================================================================

def test_evaluator_tracker_evaluation_with_subdirectories():
    """Test that tracker works when work_dir has subdirectories."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create subdirectories (should be ignored)
        os.makedirs(os.path.join(work_dir, "some_subdir"))
        os.makedirs(os.path.join(work_dir, "epoch_0"))  # This should be ignored for evaluators
        
        # Create evaluation file
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        tracker = EvaluatorTracker(work_dir)
        progress = tracker.get_progress()
        
        # Should still detect as complete evaluator
        assert progress.runner_type == 'evaluator'
        assert progress.completed_epochs == 1
        assert progress.progress_percentage == 100.0


# ============================================================================
# TESTS FOR EvaluatorTracker - INTEGRATION
# ============================================================================

def test_evaluator_tracker_with_config():
    """Test evaluator tracker with various config values."""
    with tempfile.TemporaryDirectory() as work_dir:
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Config with various fields (should not affect evaluator)
        config = {
            'epochs': 100,  # Should be ignored for evaluators
            'model': 'some_model',
            'dataset': 'some_dataset'
        }
        
        tracker = EvaluatorTracker(work_dir, config)
        progress = tracker.get_progress()
        
        # Config should not affect evaluator progress
        assert progress.total_epochs == 1  # Always 1 for evaluators, regardless of config
        assert progress.completed_epochs == 1
        assert progress.progress_percentage == 100.0
