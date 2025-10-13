from typing import List, Dict, Any
import os
import tempfile
from datetime import datetime, timedelta
import pytest


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_agent_log():
    """Create a temporary agent log file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def sample_log_entries():
    """Sample log entries for testing parsing."""
    now = datetime.now()
    recent_time = now - timedelta(hours=1)  # Use recent time within yesterday

    return [
        # Stuck process removal
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} The following processes will be killed {{config1.py: (server1, 12345), config2.py: (server2, 67890)}}",
        # Outdated cleanup
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} The following runs has not been updated in the last 30 days and will be removed",
        # Error message
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} Please fix model_v3.py. error_log='CUDA out of memory: Tried to allocate 2.00 GiB'",
        # SSH job launch
        f"{now.strftime('%Y-%m-%d %H:%M:%S')} ssh user@server1 'cd /path && python main.py --config-filepath configs/exp/baseline.py'",
        # Critical error
        f"{recent_time.strftime('%Y-%m-%d %H:%M:%S')} SSH connection failed to server2: Connection timeout",
        # Entry without timestamp (should be ignored)
        "This line has no timestamp and should be ignored",
        # Entry too old (should be filtered out)
        f"{(datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M:%S')} Old entry that should be filtered out",
    ]
