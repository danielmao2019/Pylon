import pytest
import torch
from debuggers.base_debugger import BaseDebugger


def test_base_debugger_is_abstract():
    """Test that BaseDebugger cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseDebugger()


def test_dummy_debugger_initialization(dummy_debugger):
    """Test that dummy debugger initializes correctly."""
    assert hasattr(dummy_debugger, 'output_key')
    assert dummy_debugger.output_key == "test_stats"


def test_dummy_debugger_call_basic(dummy_debugger, sample_datapoint):
    """Test basic functionality of dummy debugger."""
    result = dummy_debugger(sample_datapoint)

    # Check output structure
    assert isinstance(result, dict)
    assert "test_stats" in result

    # Check statistics content
    stats = result["test_stats"]
    assert isinstance(stats, dict)
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats

    # Check that all values are floats
    for key, value in stats.items():
        assert isinstance(value, float)


def test_dummy_debugger_different_outputs(dummy_debugger):
    """Test dummy debugger with different output tensors."""
    # Test with different shapes and values
    test_cases = [
        torch.zeros(2, 10),
        torch.ones(2, 10),
        torch.randn(2, 10) * 100,
    ]

    for outputs in test_cases:
        datapoint = {
            'inputs': torch.randn(2, 3, 32, 32),
            'labels': torch.randint(0, 10, (2,)),
            'outputs': outputs,
            'meta_info': {'idx': [0]}
        }

        result = dummy_debugger(datapoint)
        stats = result["test_stats"]

        # Verify statistics make sense
        assert stats['min'] <= stats['mean'] <= stats['max']
        assert stats['std'] >= 0


def test_another_dummy_debugger_initialization(another_dummy_debugger):
    """Test that another dummy debugger initializes correctly."""
    assert hasattr(another_dummy_debugger, 'output_key')
    assert another_dummy_debugger.output_key == "input_analysis"


def test_another_dummy_debugger_call(another_dummy_debugger, sample_datapoint):
    """Test another dummy debugger functionality."""
    result = another_dummy_debugger(sample_datapoint)

    # Check output structure
    assert isinstance(result, dict)
    assert "input_analysis" in result

    # Check analysis content
    analysis = result["input_analysis"]
    assert isinstance(analysis, dict)
    assert "input_shape" in analysis
    assert "input_mean" in analysis

    # Check values
    expected_shape = list(sample_datapoint['inputs'].shape)
    assert analysis['input_shape'] == expected_shape
    assert isinstance(analysis['input_mean'], float)


def test_debugger_with_different_output_keys():
    """Test debuggers with custom output keys."""
    # Define locally to avoid import issues
    from debuggers.base_debugger import BaseDebugger

    class CustomDummyDebugger(BaseDebugger):
        def __init__(self, output_key: str = "dummy_stats"):
            self.output_key = output_key

        def __call__(self, datapoint):
            outputs = datapoint['outputs']
            stats = {
                'mean': torch.mean(outputs).item(),
                'std': torch.std(outputs).item(),
                'min': torch.min(outputs).item(),
                'max': torch.max(outputs).item(),
            }
            return {self.output_key: stats}

    custom_debugger = CustomDummyDebugger(output_key="custom_key")

    datapoint = {
        'inputs': torch.randn(2, 3, 32, 32),
        'labels': torch.randint(0, 10, (2,)),
        'outputs': torch.randn(2, 10),
        'meta_info': {'idx': [0]}
    }

    result = custom_debugger(datapoint)
    assert "custom_key" in result
    assert len(result) == 1


def test_debugger_edge_cases():
    """Test debugger behavior with edge cases."""
    # Define locally to avoid import issues
    from debuggers.base_debugger import BaseDebugger

    class EdgeCaseDummyDebugger(BaseDebugger):
        def __init__(self, output_key: str = "dummy_stats"):
            self.output_key = output_key

        def __call__(self, datapoint):
            outputs = datapoint['outputs']
            stats = {
                'mean': torch.mean(outputs).item(),
                'std': torch.std(outputs).item(),
                'min': torch.min(outputs).item(),
                'max': torch.max(outputs).item(),
            }
            return {self.output_key: stats}

    debugger = EdgeCaseDummyDebugger()

    # Test with very small outputs
    small_datapoint = {
        'inputs': torch.randn(1, 3, 32, 32),
        'labels': torch.randint(0, 10, (1,)),
        'outputs': torch.tensor([[0.001]]),
        'meta_info': {'idx': [0]}
    }

    result = debugger(small_datapoint)
    stats = result["dummy_stats"]
    assert all(isinstance(v, float) for v in stats.values())

    # Test with large outputs
    large_datapoint = {
        'inputs': torch.randn(1, 3, 32, 32),
        'labels': torch.randint(0, 10, (1,)),
        'outputs': torch.randn(1, 1000) * 1000,
        'meta_info': {'idx': [0]}
    }

    result = debugger(large_datapoint)
    stats = result["dummy_stats"]
    assert all(isinstance(v, float) for v in stats.values())
