"""
Test tracker_factory functionality - INVALID CASES (pytest.raises).

Following CLAUDE.md testing patterns:
- Invalid input testing with exception verification
- Edge case testing
"""
import os
import tempfile
import pytest
from agents.manager.manager import Manager


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_create_tracker_config_epochs_field():
    """Test factory fails fast when config lacks 'runner' key."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, no file patterns, no runner class - invalid config
        
        config = {
            'epochs': 100,
            'model': 'test_model'
        }
        
        # Should fail fast with clear assertion error
        with pytest.raises(AssertionError, match="Config must have 'runner' key"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config)


def test_create_tracker_fail_fast_no_patterns():
    """Test that factory fails fast when no patterns match."""
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


def test_create_tracker_fail_fast_nonexistent_directory():
    """Test fail fast behavior with nonexistent directory."""
    nonexistent_dir = "/this/path/does/not/exist"
    
    with pytest.raises(ValueError) as exc_info:
        m = Manager(config_files=[], epochs=1, system_monitors={})
        m._detect_runner_type(nonexistent_dir, None)
    
    error_msg = str(exc_info.value)
    assert "Unable to determine runner type" in error_msg
    assert nonexistent_dir in error_msg


def test_create_tracker_invalid_config():
    """Test factory with invalid config that doesn't help detection."""
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


def test_create_tracker_config_string_class_name():
    """Test factory when runner class is a string instead of actual class."""
    with tempfile.TemporaryDirectory() as work_dir:
        # Empty directory, rely on config
        
        config = {
            'runner': 'BaseEvaluator'  # String instead of class (direct reference)
        }
        
        with pytest.raises(AssertionError, match="Expected runner to be a class"):
            m = Manager(config_files=[], epochs=1, system_monitors={})
            m._detect_runner_type(work_dir, config)
