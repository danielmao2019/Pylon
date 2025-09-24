"""
Test runner_detection functionality for fail-fast runner type detection.

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
from agents.manager.manager import Manager


# ============================================================================
# VALID TESTS - SUCCESSFUL DETECTION
# ============================================================================

def test_detect_runner_type_evaluator_pattern():
    """Test detection of evaluator based on evaluation_scores.json file."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluation_scores.json (evaluator pattern)
        eval_scores = {
            "aggregated": {"accuracy": 0.95},
            "per_datapoint": {"accuracy": [0.9, 0.95, 0.98]}
        }
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        runner_type = m._detect_runner_type(work_dir, None)
        assert runner_type == 'evaluator'


def test_detect_runner_type_trainer_pattern(create_epoch_files):
    """Test detection of trainer based on epoch_0/validation_scores.json structure."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create trainer pattern: epoch_0/validation_scores.json
        create_epoch_files(work_dir, 0)
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        runner_type = m._detect_runner_type(work_dir, None)
        assert runner_type == 'trainer'


def test_detect_runner_type_evaluator_takes_precedence(create_epoch_files):
    """Test that evaluator pattern takes precedence over trainer pattern."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create both patterns
        create_epoch_files(work_dir, 0)  # Trainer pattern
        
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.8, 0.9, 0.95]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)  # Evaluator pattern
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        runner_type = m._detect_runner_type(work_dir, None)
        assert runner_type == 'evaluator'  # Evaluator wins


def test_detect_runner_type_config_evaluator_class():
    """Test detection based on config runner class containing 'Evaluator'."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns
        
        # Mock config with Evaluator class (direct class reference)
        mock_evaluator_class = type('BaseEvaluator', (), {})
        config = {
            'runner': mock_evaluator_class
        }
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        runner_type = m._detect_runner_type(work_dir, config)
        assert runner_type == 'evaluator'


def test_detect_runner_type_config_trainer_class():
    """Test detection based on config runner class containing 'Trainer'."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns
        
        # Mock config with Trainer class (direct class reference)
        mock_trainer_class = type('BaseTrainer', (), {})
        config = {
            'runner': mock_trainer_class
        }
        
        m = Manager(config_files=[], epochs=1, system_monitors={})
        runner_type = m._detect_runner_type(work_dir, config)
        assert runner_type == 'trainer'


def test_detect_runner_type_deterministic():
    """Test that detection is deterministic across multiple calls."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Create evaluator pattern
        eval_scores = {"aggregated": {"acc": 0.9}, "per_datapoint": {"acc": [0.9]}}
        with open(os.path.join(work_dir, "evaluation_scores.json"), 'w') as f:
            json.dump(eval_scores, f)
        
        # Run multiple times
        m = Manager(config_files=[], epochs=1, system_monitors={})
        results = [m._detect_runner_type(work_dir, None) for _ in range(5)]
        
        # All results should be the same
        assert all(r == 'evaluator' for r in results)
        assert len(set(results)) == 1  # Only one unique result


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_detect_runner_type_config_epochs_field():
    """Test detection fails fast when config lacks 'runner' key."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns, no runner class - should fail
        
        config = {
            'epochs': 100,
            'some_other_field': 'value'
        }
        
        # Should fail fast with clear assertion error
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config)


def test_detect_runner_type_fail_fast_no_patterns():
    """Test that detection fails fast with clear error when no patterns match."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory with some irrelevant files
        with open(os.path.join(work_dir, "some_file.txt"), 'w') as f:
            f.write("irrelevant content")
        
        with pytest.raises(ValueError) as exc_info:
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, None)
        
        error_msg = str(exc_info.value)
        assert "Unable to determine runner type" in error_msg
        assert work_dir in error_msg
        # Current API does not include file listings/patterns in the message


def test_detect_runner_type_fail_fast_nonexistent_directory():
    """Test fail fast behavior with nonexistent directory."""
    nonexistent_dir = "/this/path/does/not/exist"
    
    with pytest.raises(ValueError) as exc_info:
        m = Manager(config_files=[], epochs=1, system_monitors={})
        m._detect_runner_type(nonexistent_dir, None)
    
    error_msg = str(exc_info.value)
    assert "Unable to determine runner type" in error_msg
    assert nonexistent_dir in error_msg


def test_detect_runner_type_fail_fast_with_config_info():
    """Test detection fails fast when config lacks 'runner' key."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {
            'model': 'some_model',
            'dataset': 'some_dataset'
        }
        
        # Should fail fast with assertion error since no 'runner' key
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config)


def test_detect_runner_type_epoch_0_exists_but_no_validation_scores():
    """Test that epoch_0 directory without validation_scores.json is not detected as trainer."""
    with tempfile.TemporaryDirectory() as work_dir:
        epoch_0_dir = os.path.join(work_dir, "epoch_0")
        os.makedirs(epoch_0_dir)
        
        # Create other files but not validation_scores.json
        with open(os.path.join(epoch_0_dir, "some_other_file.txt"), 'w') as f:
            f.write("content")
        
        with pytest.raises(ValueError) as exc_info:
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, None)
        
        assert "Unable to determine runner type" in str(exc_info.value)


def test_detect_runner_type_invalid_config_runner_class():
    """Test detection with invalid runner class in config."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Config with runner class that doesn't contain Trainer or Evaluator (direct class reference)
        mock_other_class = type('SomeOtherClass', (), {})
        config = {
            'runner': mock_other_class
        }
        
        with pytest.raises(ValueError) as exc_info:
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config)
        
        assert "Unable to determine runner type" in str(exc_info.value)


@pytest.mark.parametrize("config_variant", [
    {},  # Empty config
    None,  # No config
])
def test_detect_runner_type_various_invalid_configs(config_variant):
    """Test detection behavior with various config variants that result in ValueError."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty work directory
        
        with pytest.raises(ValueError) as exc_info:
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config_variant)
        
        assert "Unable to determine runner type" in str(exc_info.value)


@pytest.mark.parametrize("config_variant", [
    {'some_field': 'value'},  # Config without runner key
])
def test_detect_runner_type_configs_missing_runner_key(config_variant):
    """Test detection behavior with configs missing runner key (AssertionError)."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty work directory
        
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config_variant)


@pytest.mark.parametrize("config_variant", [
    {'runner': {}},  # Config with empty runner
    {'runner': {'class': None}},  # Config with None runner class
])
def test_detect_runner_type_configs_invalid_runner_type(config_variant):
    """Test detection behavior with configs where runner is not a class (AssertionError)."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty work directory
        
        with pytest.raises(AssertionError, match="Expected runner to be a class"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config_variant)


def test_detect_runner_type_config_string_class_name():
    """Test detection when runner class is a string instead of actual class."""
    with tempfile.TemporaryDirectory() as work_dir:
        config = {
            'runner': 'BaseEvaluator'  # String instead of class (direct reference)
        }
        
        with pytest.raises(AssertionError, match="Expected runner to be a class"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config)
