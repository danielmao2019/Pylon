import pytest
import torch
import torch.nn as nn
from debuggers.wrappers.sequential_debugger import SequentialDebugger
from debuggers.base_debugger import BaseDebugger


class MockDebugger(BaseDebugger):
    """Simple debugger for testing enabled/disabled state."""
    
    def __call__(self, datapoint, model):
        return {'test_output': 'debugger_called'}


class MockModel(nn.Module):
    """Simple test model."""
    
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)


def test_debugger_returns_empty_when_disabled():
    """Test that debugger returns empty dict when disabled."""
    model = MockModel()
    
    debuggers_config = [
        {
            'name': 'test_debugger',
            'debugger_config': {
                'class': MockDebugger,
                'args': {}
            }
        }
    ]
    
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=model,
        page_size_mb=1
    )
    
    # Create test datapoint
    datapoint = {
        'inputs': torch.randn(1, 10, dtype=torch.float32),
        'outputs': torch.randn(1, 5, dtype=torch.float32),
        'meta_info': {'idx': [0]}
    }
    
    # Test disabled state
    debugger.enabled = False
    result = debugger(datapoint, model)
    assert result == {}
    
    # Test enabled state
    debugger.enabled = True
    result = debugger(datapoint, model)
    assert result != {}
    assert 'test_debugger' in result


def test_debugger_enabled_disabled_state_management():
    """Test proper state management of enabled/disabled flag."""
    model = MockModel()
    
    debuggers_config = [
        {
            'name': 'test_debugger',
            'debugger_config': {
                'class': MockDebugger,
                'args': {}
            }
        }
    ]
    
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=model,
        page_size_mb=1
    )
    
    # Default state should be disabled
    assert debugger.enabled is False
    
    # Can enable
    debugger.enabled = True
    assert debugger.enabled is True
    
    # Can disable
    debugger.enabled = False
    assert debugger.enabled is False


def test_debugger_buffer_not_filled_when_disabled():
    """Test that buffer is not filled when debugger is disabled."""
    model = MockModel()
    
    debuggers_config = [
        {
            'name': 'test_debugger',
            'debugger_config': {
                'class': MockDebugger,
                'args': {}
            }
        }
    ]
    
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=model,
        page_size_mb=1
    )
    
    datapoint = {
        'inputs': torch.randn(1, 10, dtype=torch.float32),
        'outputs': torch.randn(1, 5, dtype=torch.float32),
        'meta_info': {'idx': [0]}
    }
    
    # Ensure disabled
    debugger.enabled = False
    
    # Call debugger
    result = debugger(datapoint, model)
    
    # Wait for any async operations
    debugger._buffer_queue.join()
    
    # Buffer should remain empty
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == 0
        assert debugger.current_page_size == 0
