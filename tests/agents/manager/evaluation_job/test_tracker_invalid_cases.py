"""
Evaluator detection behavior - INVALID CASES (pytest.raises).

Aligned to Manager._detect_runner_type API.
"""
import pytest
from agents.manager.manager import Manager


# ============================================================================
# INVALID TESTS - EXPECTED FAILURES (pytest.raises)
# ============================================================================

def test_evaluator_detection_nonexistent_work_dir():
    """Nonexistent directory should raise informative ValueError."""
    nonexistent_dir = "/this/path/does/not/exist"
    m = Manager(commands=[], epochs=1, system_monitors={})
    with pytest.raises(ValueError) as exc_info:
        m._detect_runner_type(nonexistent_dir, None)
    assert "Unable to determine runner type" in str(exc_info.value)


def test_evaluator_detection_invalid_work_dir_type():
    """Invalid work_dir type should error via path handling."""
    m = Manager(commands=[], epochs=1, system_monitors={})
    with pytest.raises(Exception):
        m._detect_runner_type(123, None)  # type: ignore[arg-type]
