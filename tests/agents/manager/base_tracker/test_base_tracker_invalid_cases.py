"""
Test BaseTracker functionality - INVALID CASES (pytest.raises).

Following CLAUDE.md testing patterns:
- Invalid input testing with exception verification
- Abstract method contract testing
"""
import os
import pytest
from agents.manager.manager import Manager


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_manager_detect_trainer_input_validation():
    """Validate Manager._detect_runner_type input handling for trainer case."""
    m = Manager(config_files=[], epochs=1, system_monitors={})
    # Non-string path should raise due to os.path functions; simulate by expecting ValueError
    with pytest.raises(Exception):
        m._detect_runner_type(123, None)  # type: ignore[arg-type]
    # Nonexistent directory yields ValueError with informative message
    nonexistent_dir = "/this/path/does/not/exist"
    with pytest.raises(ValueError) as exc_info:
        m._detect_runner_type(nonexistent_dir, None)
    assert "Unable to determine runner type" in str(exc_info.value)


def test_manager_detect_evaluator_input_validation():
    """Validate Manager._detect_runner_type input handling for evaluator case."""
    m = Manager(config_files=[], epochs=1, system_monitors={})
    with pytest.raises(Exception):
        m._detect_runner_type(123, None)  # type: ignore[arg-type]
    nonexistent_dir = "/this/path/does/not/exist"
    with pytest.raises(ValueError) as exc_info:
        m._detect_runner_type(nonexistent_dir, None)
    assert "Unable to determine runner type" in str(exc_info.value)


def test_manager_detect_requires_artifacts_or_config():
    """Manager requires expected artifacts or runner class in config."""
    m = Manager(config_files=[], epochs=1, system_monitors={})
    with pytest.raises(ValueError):
        with pytest.TempdirFactory():
            pass
    # Use a real temp dir without artifacts
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(ValueError):
            m._detect_runner_type(d, None)
