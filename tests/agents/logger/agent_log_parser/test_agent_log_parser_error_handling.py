import os
import pytest
from agents.logger import AgentLogParser


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_log_parsing_error_handling(temp_agent_log):
    """Test error handling when log file becomes corrupted during reading."""
    from datetime import datetime
    
    with open(temp_agent_log, 'w') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Normal entry\n")
    
    # Make file unreadable after creation
    os.chmod(temp_agent_log, 0o000)
    
    try:
        parser = AgentLogParser(agent_log_path=temp_agent_log)
        
        # Should raise PermissionError when trying to read
        with pytest.raises(PermissionError):
            parser.extract_key_events_since_yesterday()
            
    finally:
        # Restore permissions for cleanup
        os.chmod(temp_agent_log, 0o644)


def test_invalid_timestamp_format_error(temp_agent_log):
    """Test that invalid timestamp format raises ValueError."""
    with open(temp_agent_log, 'w') as f:
        f.write("2024-13-45 25:99:99 Invalid timestamp format\n")
    
    parser = AgentLogParser(agent_log_path=temp_agent_log)
    
    # Should raise ValueError for invalid timestamp
    with pytest.raises(ValueError):
        parser.extract_key_events_since_yesterday()
