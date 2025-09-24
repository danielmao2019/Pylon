"""
Shared test fixtures and helper functions for agent tests.

This conftest.py provides common test data creation functions and import handling
for agent-related tests.
"""
import os
import tempfile
import sys
from pathlib import Path
import pytest

# ============================================================================
# COMMON TEST CONSTANTS
# ============================================================================

# Standard expected files for most tests
EXPECTED_FILES = ["training_losses.pt", "optimizer_buffer.json", "validation_scores.json"]

# ============================================================================
# PYTEST FIXTURES
# ============================================================================

@pytest.fixture
def temp_agent_log():
    """Create a temporary agent log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_config_files():
    """Sample config files for testing."""
    return [
        "configs/exp/baseline.py",
        "configs/exp/model_v2.py",
        "configs/exp/ablation.py"
    ]


@pytest.fixture
def sample_expected_files():
    """Sample expected files for testing."""
    return ["train_metrics.json", "val_metrics.json", "model.pt"]
