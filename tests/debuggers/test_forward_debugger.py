import pytest
import torch
import torch.nn as nn
from conftest import FeatureMapDebugger, ActivationStatsDebugger, LayerOutputDebugger


@pytest.mark.parametrize("layer_name", ["conv1", "relu", "conv2", "pool", "fc"])
def test_feature_map_debugger_initialization(layer_name):
    """Test FeatureMapDebugger initialization with different layer names."""
    debugger = FeatureMapDebugger(layer_name=layer_name)
    assert debugger.layer_name == layer_name
    assert debugger.last_capture is None


@pytest.mark.parametrize("batch_size,channels,height,width", [
    (1, 16, 32, 32),
    (2, 32, 64, 64), 
    (4, 64, 16, 16)
])
def test_feature_map_debugger_process_forward(batch_size, channels, height, width):
    """Test FeatureMapDebugger process_forward with different tensor sizes."""
    debugger = FeatureMapDebugger(layer_name="test_layer")
    
    # Create mock input and output
    mock_module = nn.Conv2d(3, channels, 3)
    mock_input = (torch.randn(batch_size, 3, height, width),)
    mock_output = torch.randn(batch_size, channels, height, width)
    
    # Call process_forward
    result = debugger.process_forward(mock_module, mock_input, mock_output)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'feature_map' in result
    assert 'stats' in result
    assert 'layer_name' in result
    
    # Check stats
    stats = result['stats']
    assert stats['shape'] == [batch_size, channels, height, width]
    assert result['layer_name'] == "test_layer"


def test_feature_map_debugger_forward_hook():
    """Test FeatureMapDebugger with forward hook."""
    debugger = FeatureMapDebugger(layer_name="conv1")
    
    # Create mock data
    mock_module = nn.Conv2d(3, 16, 3)
    mock_input = (torch.randn(2, 3, 32, 32),)
    mock_output = torch.randn(2, 16, 32, 32)
    
    # Call forward hook
    debugger.forward_hook_fn(mock_module, mock_input, mock_output)
    
    # Check that data was captured
    assert debugger.last_capture is not None
    assert 'feature_map' in debugger.last_capture
    assert 'stats' in debugger.last_capture


def test_feature_map_debugger_call(sample_datapoint, dummy_model):
    """Test FeatureMapDebugger __call__ method."""
    debugger = FeatureMapDebugger(layer_name="conv1")
    
    # Without any captured data
    result = debugger(sample_datapoint, dummy_model)
    assert result is None
    
    # With captured data
    debugger.last_capture = {'test': 'data'}
    result = debugger(sample_datapoint, dummy_model)
    assert result == {'test': 'data'}


@pytest.mark.parametrize("layer_name", ["relu", "conv1", "bn1"])
def test_activation_stats_debugger_initialization(layer_name):
    """Test ActivationStatsDebugger initialization."""
    debugger = ActivationStatsDebugger(layer_name=layer_name)
    assert debugger.layer_name == layer_name
    assert debugger.last_capture is None


def test_activation_stats_debugger_process_forward():
    """Test ActivationStatsDebugger process_forward method."""
    debugger = ActivationStatsDebugger(layer_name="relu")
    
    # Create mock data
    mock_module = nn.ReLU()
    mock_input = (torch.randn(2, 16, 32, 32),)
    mock_output = torch.relu(mock_input[0])  # Ensure positive values for ReLU
    
    # Call process_forward
    result = debugger.process_forward(mock_module, mock_input, mock_output)
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'layer_name' in result
    assert 'module_type' in result
    assert 'activation_stats' in result
    assert 'sample_values' in result
    
    # Check stats
    stats = result['activation_stats']
    required_stats = ['shape', 'mean', 'std', 'sparsity', 'positive_ratio']
    for stat in required_stats:
        assert stat in stats
    
    # Check values
    assert result['layer_name'] == "relu"
    assert result['module_type'] == "ReLU"
    assert len(result['sample_values']) <= 100


@pytest.mark.parametrize("downsample_factor", [1, 2, 4])
def test_layer_output_debugger_initialization(downsample_factor):
    """Test LayerOutputDebugger initialization with different downsample factors."""
    debugger = LayerOutputDebugger(layer_name="conv2", downsample_factor=downsample_factor)
    assert debugger.layer_name == "conv2"
    assert debugger.downsample_factor == downsample_factor
    assert debugger.last_capture is None


@pytest.mark.parametrize("downsample_factor,expected_downsampled", [
    (1, False),
    (2, True),
    (4, True)
])
def test_layer_output_debugger_process_forward(downsample_factor, expected_downsampled):
    """Test LayerOutputDebugger process_forward with different downsample factors."""
    debugger = LayerOutputDebugger(layer_name="conv2", downsample_factor=downsample_factor)
    
    # Create mock data
    mock_module = nn.Conv2d(16, 32, 3)
    mock_input = (torch.randn(2, 16, 32, 32),)
    mock_output = torch.randn(2, 32, 32, 32)
    
    # Call process_forward
    result = debugger.process_forward(mock_module, mock_input, mock_output)
    
    # Check result structure
    assert isinstance(result, dict)
    required_keys = ['layer_name', 'original_shape', 'output', 'downsampled', 'downsample_factor']
    for key in required_keys:
        assert key in result
    
    # Check values
    assert result['layer_name'] == "conv2"
    assert result['original_shape'] == [2, 32, 32, 32]
    assert result['downsample_factor'] == downsample_factor
    assert result['downsampled'] is expected_downsampled
    
    # Check output shape
    if expected_downsampled:
        expected_shape = [2, 32, 32 // downsample_factor, 32 // downsample_factor]
        assert list(result['output'].shape) == expected_shape
    else:
        assert list(result['output'].shape) == result['original_shape']


@pytest.mark.parametrize("output_type", ["tensor", "non_tensor"])
def test_forward_debugger_non_tensor_output(output_type):
    """Test forward debuggers with different output types."""
    debugger = FeatureMapDebugger(layer_name="test")
    
    # Create mock data
    mock_module = nn.Identity()
    mock_input = (torch.randn(2, 3, 32, 32),)
    
    if output_type == "tensor":
        mock_output = torch.randn(2, 3, 32, 32)
        result = debugger.process_forward(mock_module, mock_input, mock_output)
        assert 'feature_map' in result
        assert 'stats' in result
    else:
        mock_output = "not a tensor"
        result = debugger.process_forward(mock_module, mock_input, mock_output)
        assert isinstance(result, dict)
        assert 'error' in result
        assert result['error'] == 'Output is not a tensor'


def test_forward_debugger_memory_cleanup():
    """Test that forward debuggers properly clean up memory."""
    debugger = FeatureMapDebugger(layer_name="test")
    
    # Create large tensors
    large_output = torch.randn(10, 256, 64, 64)
    mock_module = nn.Conv2d(3, 256, 3)
    mock_input = (torch.randn(10, 3, 64, 64),)
    
    # Call forward hook to set last_capture
    debugger.forward_hook_fn(mock_module, mock_input, large_output)
    
    # Verify data was captured and is on CPU
    assert debugger.last_capture is not None
    assert debugger.last_capture['feature_map'].device.type == 'cpu'


def test_multiple_forward_debuggers():
    """Test using multiple forward debuggers."""
    debugger1 = FeatureMapDebugger(layer_name="conv1")
    debugger2 = ActivationStatsDebugger(layer_name="conv1")
    
    # Create mock data
    mock_module = nn.Conv2d(3, 16, 3)
    mock_input = (torch.randn(2, 3, 32, 32),)
    mock_output = torch.randn(2, 16, 32, 32)
    
    # Process with both debuggers
    result1 = debugger1.process_forward(mock_module, mock_input, mock_output)
    result2 = debugger2.process_forward(mock_module, mock_input, mock_output)
    
    # Check that both produce different but valid results
    assert 'feature_map' in result1
    assert 'activation_stats' in result2
    assert result1['layer_name'] == result2['layer_name'] == "conv1"
