import pytest
import torch
from debuggers.wrappers.sequential_debugger import SequentialDebugger


def test_sequential_debugger_call_basic(sequential_debugger_basic, sample_datapoint, dummy_model):
    """Test basic __call__ functionality."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Call debugger
    result = debugger(sample_datapoint, dummy_model)
    
    # Should return dict with debugger outputs
    assert isinstance(result, dict)
    assert len(result) == 2  # Two debuggers in basic config
    assert 'dummy_stats' in result
    assert 'input_analysis' in result


def test_sequential_debugger_call_disabled(sequential_debugger_basic, sample_datapoint, dummy_model):
    """Test __call__ when debugger is disabled."""
    debugger = sequential_debugger_basic
    debugger.enabled = False
    
    # Call debugger when disabled
    result = debugger(sample_datapoint, dummy_model)
    
    # Should return empty dict
    assert result == {}


def test_sequential_debugger_call_empty_config(sequential_debugger_empty, sample_datapoint, dummy_model):
    """Test __call__ with empty debugger config."""
    debugger = sequential_debugger_empty
    debugger.enabled = True
    
    # Call debugger with no configured debuggers
    result = debugger(sample_datapoint, dummy_model)
    
    # Should return empty dict
    assert result == {}


def test_sequential_debugger_with_forward_hooks(sequential_debugger_forward_hooks, sample_datapoint, dummy_model):
    """Test debugger with forward hook debuggers."""
    debugger = sequential_debugger_forward_hooks
    debugger.enabled = True
    
    # Run a forward pass through the model to trigger hooks
    with torch.no_grad():
        _ = dummy_model(sample_datapoint['inputs'])
    
    # Call debugger (should include forward hook data)
    result = debugger(sample_datapoint, dummy_model)
    
    # Should have both regular and forward debugger outputs
    assert isinstance(result, dict)
    assert 'dummy_stats' in result  # Regular debugger
    assert 'conv2_features' in result  # Forward hook debugger


def test_sequential_debugger_model_parameter_passing(debuggers_config, dummy_model):
    """Test that model parameter is correctly passed to child debuggers."""
    # Create a custom debugger that checks if model parameter is passed
    from debuggers.base_debugger import BaseDebugger
    
    class ModelCheckDebugger(BaseDebugger):
        def __call__(self, datapoint, model):
            # Verify model is passed and is correct type
            assert model is not None
            assert isinstance(model, torch.nn.Module)
            return {'model_received': True, 'model_type': type(model).__name__}
    
    # Create config with model-checking debugger
    config_with_model_check = [
        {
            'name': 'model_checker',
            'debugger_config': {
                'class': ModelCheckDebugger,
                'args': {}
            }
        }
    ]
    
    debugger = SequentialDebugger(
        debuggers_config=config_with_model_check,
        model=dummy_model,
        page_size_mb=1
    )
    debugger.enabled = True
    
    sample_datapoint = {
        'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
        'outputs': torch.randn(1, 10, dtype=torch.float32),
        'meta_info': {'idx': [0]}
    }
    
    # Call debugger
    result = debugger(sample_datapoint, dummy_model)
    
    # Check model was passed correctly
    assert result['model_checker']['model_received'] is True
    assert result['model_checker']['model_type'] == 'DummyModel'


def test_sequential_debugger_data_flow_integration(sequential_debugger_basic, sample_datapoint, dummy_model):
    """Test complete data flow from __call__ to buffer."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Call debugger
    result = debugger(sample_datapoint, dummy_model)
    
    # Wait for buffer processing
    debugger._buffer_queue.join()
    
    # Check data flow: call -> debuggers -> buffer
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Check data was added to buffer
    with debugger._buffer_lock:
        assert len(debugger.current_page_data) == 1
        assert 0 in debugger.current_page_data
        
        # Check buffer contains the same data structure
        buffered_data = debugger.current_page_data[0]
        assert set(buffered_data.keys()) == set(result.keys())


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_sequential_debugger_different_batch_sizes(debuggers_config, dummy_model, batch_size):
    """Test debugger works with different batch sizes."""
    debugger = SequentialDebugger(
        debuggers_config=debuggers_config,
        model=dummy_model,
        page_size_mb=1
    )
    debugger.enabled = True
    
    # Create datapoint with specific batch size
    datapoint = {
        'inputs': torch.randn(batch_size, 3, 32, 32, dtype=torch.float32),
        'outputs': torch.randn(batch_size, 10, dtype=torch.float32),
        'meta_info': {'idx': [0]}
    }
    
    # Call debugger
    result = debugger(datapoint, dummy_model)
    
    # Should work regardless of batch size
    assert isinstance(result, dict)
    assert len(result) == 2  # Two debuggers in config


def test_sequential_debugger_tensor_device_handling(sequential_debugger_basic, dummy_model):
    """Test debugger handles tensors on different devices."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    # Create datapoint with mixed device tensors
    if torch.cuda.is_available():
        inputs = torch.randn(1, 3, 32, 32, dtype=torch.float32).cuda()
        outputs = torch.randn(1, 10, dtype=torch.float32).cuda()
    else:
        inputs = torch.randn(1, 3, 32, 32, dtype=torch.float32)
        outputs = torch.randn(1, 10, dtype=torch.float32)
    
    datapoint = {
        'inputs': inputs,
        'outputs': outputs,
        'meta_info': {'idx': [0]}
    }
    
    # Call debugger
    result = debugger(datapoint, dummy_model)
    
    # Should handle device differences gracefully
    assert isinstance(result, dict)
    assert len(result) > 0
    
    # Wait for buffer processing and check CPU conversion
    debugger._buffer_queue.join()
    with debugger._buffer_lock:
        buffered_data = debugger.current_page_data[0]
        # All buffered tensors should be on CPU due to apply_tensor_op
        for debugger_name, debugger_output in buffered_data.items():
            if isinstance(debugger_output, dict):
                for key, value in debugger_output.items():
                    if isinstance(value, torch.Tensor):
                        assert value.device.type == 'cpu'


def test_sequential_debugger_error_propagation(debuggers_config, dummy_model):
    """Test that debugger errors are properly handled."""
    # Create a debugger that raises an exception
    from debuggers.base_debugger import BaseDebugger
    
    class ErrorDebugger(BaseDebugger):
        def __call__(self, datapoint, model):
            raise ValueError("Test error from debugger")
    
    class GoodDebugger(BaseDebugger):
        def __call__(self, datapoint, model):
            return {'good_output': 'success'}
    
    config_with_error = [
        {
            'name': 'good_debugger',
            'debugger_config': {
                'class': GoodDebugger,
                'args': {}
            }
        },
        {
            'name': 'error_debugger',
            'debugger_config': {
                'class': ErrorDebugger,
                'args': {}
            }
        }
    ]
    
    debugger = SequentialDebugger(
        debuggers_config=config_with_error,
        model=dummy_model,
        page_size_mb=1
    )
    debugger.enabled = True
    
    sample_datapoint = {
        'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
        'outputs': torch.randn(1, 10, dtype=torch.float32),
        'meta_info': {'idx': [0]}
    }
    
    # Should raise the error from the error debugger
    with pytest.raises(ValueError, match="Test error from debugger"):
        debugger(sample_datapoint, dummy_model)


def test_sequential_debugger_output_structure_consistency(sequential_debugger_basic, dummy_model):
    """Test that output structure is consistent across multiple calls."""
    debugger = sequential_debugger_basic
    debugger.enabled = True
    
    results = []
    for i in range(3):
        datapoint = {
            'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
            'outputs': torch.randn(1, 10, dtype=torch.float32),
            'meta_info': {'idx': [i]}
        }
        result = debugger(datapoint, dummy_model)
        results.append(result)
    
    # Check all results have same structure
    first_keys = set(results[0].keys())
    for i, result in enumerate(results[1:], 1):
        assert set(result.keys()) == first_keys, f"Result {i} has different keys"
        
        # Check each debugger output structure
        for debugger_name in first_keys:
            assert debugger_name in result
            assert isinstance(result[debugger_name], dict)


def test_sequential_debugger_integration_with_real_model_forward(sequential_debugger_forward_hooks, dummy_model):
    """Test integration with actual model forward pass."""
    debugger = sequential_debugger_forward_hooks
    debugger.enabled = True
    
    # Create input data
    input_data = torch.randn(2, 3, 32, 32, dtype=torch.float32)
    
    # Run model forward pass (this should trigger forward hooks)
    with torch.no_grad():
        model_output = dummy_model(input_data)
    
    # Create datapoint with model outputs
    datapoint = {
        'inputs': input_data,
        'outputs': model_output,
        'meta_info': {'idx': [0]}
    }
    
    # Call debugger
    result = debugger(datapoint, dummy_model)
    
    # Should have captured forward hook data
    assert isinstance(result, dict)
    assert 'conv2_features' in result  # Forward hook debugger
    
    # Forward hook should have captured actual conv2 output
    if result['conv2_features'] is not None:  # May be None if no recent forward pass
        forward_data = result['conv2_features']
        assert isinstance(forward_data, dict)


def test_sequential_debugger_inheritance_and_interface(sequential_debugger_basic):
    """Test that SequentialDebugger properly implements BaseDebugger interface."""
    from debuggers.base_debugger import BaseDebugger
    
    # Check inheritance
    assert isinstance(sequential_debugger_basic, BaseDebugger)
    
    # Check required methods exist
    assert hasattr(sequential_debugger_basic, '__call__')
    assert callable(sequential_debugger_basic.__call__)
    
    # Check method signature
    import inspect
    call_signature = inspect.signature(sequential_debugger_basic.__call__)
    params = list(call_signature.parameters.keys())
    assert 'datapoint' in params
    assert 'model' in params
