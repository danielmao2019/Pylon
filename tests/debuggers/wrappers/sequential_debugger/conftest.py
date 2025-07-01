import pytest
import tempfile
import shutil
from debuggers.wrappers.sequential_debugger import SequentialDebugger


@pytest.fixture
def sequential_debugger_basic(debuggers_config, dummy_model):
    """Create a basic SequentialDebugger for testing."""
    return SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model,
        page_size_mb=1  # Small page size for testing
    )


@pytest.fixture
def sequential_debugger_empty(empty_debuggers_config, dummy_model):
    """Create a SequentialDebugger with no debuggers."""
    return SequentialDebugger(
        debuggers_config=empty_debuggers_config,
        model=dummy_model,
        page_size_mb=1
    )


@pytest.fixture
def sequential_debugger_forward_hooks(mixed_debuggers_config, dummy_model):
    """Create a SequentialDebugger with forward hooks."""
    return SequentialDebugger(
        debuggers_config=mixed_debuggers_config,
        model=dummy_model,
        page_size_mb=1
    )


@pytest.fixture
def enabled_sequential_debugger(sequential_debugger_basic):
    """Create an enabled SequentialDebugger."""
    sequential_debugger_basic.enabled = True
    return sequential_debugger_basic


@pytest.fixture
def temp_debugger_output_dir():
    """Create a temporary directory for debugger output files."""
    temp_dir = tempfile.mkdtemp(prefix="debugger_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)