import pytest
import torch
import torch.nn as nn
from debuggers.wrappers.sequential_debugger import SequentialDebugger


def test_sequential_debugger_basic_initialization(debuggers_config, dummy_model):
    """Test basic initialization of SequentialDebugger."""
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model,
        page_size_mb=100
    )

    # Test basic attributes
    assert hasattr(debugger, 'debuggers')
    assert hasattr(debugger, 'enabled')
    assert hasattr(debugger, 'page_size')
    assert hasattr(debugger, 'current_page_idx')
    assert hasattr(debugger, 'current_page_size')
    assert hasattr(debugger, 'current_page_data')

    # Test initial state
    assert debugger.enabled is False  # Should start disabled
    assert debugger.page_size == 100 * 1024 * 1024  # Converted to bytes
    assert debugger.current_page_idx == 0
    assert debugger.current_page_size == 0
    assert isinstance(debugger.current_page_data, dict)
    assert len(debugger.current_page_data) == 0


def test_sequential_debugger_threading_setup(debuggers_config, dummy_model):
    """Test that threading components are properly initialized."""
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model
    )

    # Test threading attributes
    assert hasattr(debugger, '_buffer_lock')
    assert hasattr(debugger, '_buffer_queue')
    assert hasattr(debugger, '_buffer_thread')

    # Test thread is running
    assert debugger._buffer_thread.is_alive()
    assert debugger._buffer_thread.daemon


def test_sequential_debugger_empty_config(empty_debuggers_config, dummy_model):
    """Test initialization with empty debugger config."""
    debugger = SequentialDebugger(
        debuggers_config=empty_debuggers_config,
        model=dummy_model
    )

    assert len(debugger.debuggers) == 0
    assert len(debugger.forward_debuggers) == 0


def test_sequential_debugger_debugger_creation(debuggers_config, dummy_model):
    """Test that debuggers are properly created from config."""
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model
    )

    # Test debuggers dictionary
    assert len(debugger.debuggers) == 2
    assert 'dummy_stats' in debugger.debuggers
    assert 'input_analysis' in debugger.debuggers

    # Test debugger types
    assert hasattr(debugger.debuggers['dummy_stats'], '__call__')
    assert hasattr(debugger.debuggers['input_analysis'], '__call__')


def test_sequential_debugger_forward_hook_tracking(mixed_debuggers_config, dummy_model):
    """Test that forward debuggers are properly tracked."""
    debugger = SequentialDebugger(
        debuggers_config=mixed_debuggers_config,
        model=dummy_model
    )

    # Should have one forward debugger
    assert len(debugger.forward_debuggers) == 1
    assert 'conv2' in debugger.forward_debuggers
    assert len(debugger.forward_debuggers['conv2']) == 1

    # Check the forward debugger is the correct instance
    forward_debugger = debugger.forward_debuggers['conv2'][0]
    assert isinstance(forward_debugger, TestFeatureMapDebugger)
    assert forward_debugger.layer_name == 'conv2'


def test_sequential_debugger_duplicate_names_error(dummy_model):
    """Test that duplicate debugger names raise an error."""
    duplicate_config = [
        {
            'name': 'same_name',
            'debugger_config': {
                'class': type('DummyDebugger', (object,), {'__call__': lambda self, x: {}}),
                'args': {}
            }
        },
        {
            'name': 'same_name',  # Duplicate name
            'debugger_config': {
                'class': type('AnotherDebugger', (object,), {'__call__': lambda self, x: {}}),
                'args': {}
            }
        }
    ]

    with pytest.raises(AssertionError, match="Duplicate debugger name"):
        SequentialDebugger(
            debuggers_config=duplicate_config,
            model=dummy_model
        )


def test_sequential_debugger_custom_page_size(debuggers_config, dummy_model):
    """Test initialization with custom page size."""
    custom_page_size = 50  # 50 MB
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model,
        page_size_mb=custom_page_size
    )

    assert debugger.page_size == custom_page_size * 1024 * 1024


def test_sequential_debugger_model_layer_validation(forward_debugger_config, dummy_model):
    """Test validation of model layers for forward debuggers."""
    # This should work - conv2 exists in dummy_model
    debugger = SequentialDebugger(
        debuggers_config=[forward_debugger_config],
        model=dummy_model
    )

    assert len(debugger.forward_debuggers) == 1
    assert 'conv2' in debugger.forward_debuggers


def test_sequential_debugger_missing_layer_warning(dummy_model, capsys):
    """Test warning when trying to hook a non-existent layer."""
    # Define test debugger locally since examples.py is moved to docs
    from debuggers.forward_debugger import ForwardDebugger
    
    class TestDebugger(ForwardDebugger):
        def process_forward(self, module, input, output):
            return {'test': 'data'}

    missing_layer_config = {
        'name': 'missing_layer',
        'debugger_config': {
            'class': TestDebugger,
            'args': {
                'layer_name': 'nonexistent_layer'
            }
        }
    }

    debugger = SequentialDebugger(
        debuggers_config=[missing_layer_config],
        model=dummy_model
    )

    # Check that warning was printed
    captured = capsys.readouterr()
    assert "Could not find layer 'nonexistent_layer'" in captured.out


def test_sequential_debugger_inheritance():
    """Test that SequentialDebugger inherits from BaseDebugger."""
    from debuggers.base_debugger import BaseDebugger

    # Check inheritance
    assert issubclass(SequentialDebugger, BaseDebugger)


def test_sequential_debugger_output_dir_initialization(debuggers_config, dummy_model):
    """Test that output_dir is initially None."""
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model
    )

    assert debugger.output_dir is None