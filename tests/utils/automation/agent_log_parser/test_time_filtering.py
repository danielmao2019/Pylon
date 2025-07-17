from datetime import datetime, timedelta
import pytest
from utils.automation.agent_log_parser import AgentLogParser


# ============================================================================
# TIME FILTERING TESTS
# ============================================================================

def test_time_based_filtering(temp_agent_log):
    """Test that only events from yesterday onward are included."""
    yesterday = datetime.now() - timedelta(days=1)
    old_date = datetime.now() - timedelta(days=3)
    
    log_entries = [
        f"{old_date.strftime('%Y-%m-%d %H:%M:%S')} Old event should be filtered",
        f"{yesterday.strftime('%Y-%m-%d %H:%M:%S')} The following processes will be killed {{config1.py: (server1, 12345)}}",
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Recent event should be included"
    ]
    
    with open(temp_agent_log, 'w') as f:
        for entry in log_entries:
            f.write(entry + '\n')
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    events = parser.extract_key_events_since_yesterday()
    
    # Should only include events from yesterday onward
    for event in events:
        assert event['timestamp'] >= yesterday