"""
Shared test fixtures and helper functions for agent tests.

This conftest.py provides common test data creation functions and import handling
for agent-related tests.
"""
from typing import List, Dict, Any
import os
import tempfile
import json
import pytest
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now we can import the agents modules
from agents.logs_snapshot import LogsSnapshot
from agents.agent_log_parser import AgentLogParser
from agents.daily_summary_generator import DailySummaryGenerator
from utils.monitor.system_monitor import SystemMonitor
from utils.automation.run_status import RunStatus
from utils.automation.run_status.session_progress import ProgressInfo
from utils.monitor.process_info import ProcessInfo

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
