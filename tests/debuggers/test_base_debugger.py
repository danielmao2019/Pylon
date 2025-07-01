from typing import Dict, Any
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


def test_dummy_debugger_call_basic(dummy_debugger, sample_datapoint, dummy_model):
    """Test basic functionality of dummy debugger."""
    result = dummy_debugger(sample_datapoint, dummy_model)

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


@pytest.mark.parametrize("output_tensor,description", [
    (torch.zeros(2, 10), "zeros"),
    (torch.ones(2, 10), "ones"),
    (torch.randn(2, 10) * 100, "large_random")
])
def test_dummy_debugger_different_outputs(dummy_debugger, dummy_model, output_tensor, description):
    """Test dummy debugger with different output tensors."""
    datapoint = {
        'inputs': torch.randn(2, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (2,), dtype=torch.int64),
        'outputs': output_tensor,
        'meta_info': {'idx': [0]}
    }

    result = dummy_debugger(datapoint, dummy_model)
    stats = result["test_stats"]

    # Verify statistics make sense
    assert stats['min'] <= stats['mean'] <= stats['max']
    assert stats['std'] >= 0


def test_another_dummy_debugger_initialization(another_dummy_debugger):
    """Test that another dummy debugger initializes correctly."""
    assert hasattr(another_dummy_debugger, 'output_key')
    assert another_dummy_debugger.output_key == "input_analysis"


def test_another_dummy_debugger_call(another_dummy_debugger, sample_datapoint, dummy_model):
    """Test another dummy debugger functionality."""
    result = another_dummy_debugger(sample_datapoint, dummy_model)

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


@pytest.mark.parametrize("output_key", ["custom_key", "special_stats", "my_debug_output"])
def test_debugger_with_different_output_keys(dummy_model, output_key):
    """Test debuggers with custom output keys."""

    class CustomDummyDebugger(BaseDebugger):
        def __init__(self, output_key: str = "dummy_stats"):
            self.output_key = output_key

        def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
            outputs = datapoint['outputs']
            stats = {
                'mean': torch.mean(outputs).item(),
                'std': torch.std(outputs).item(),
                'min': torch.min(outputs).item(),
                'max': torch.max(outputs).item(),
            }
            return {self.output_key: stats}

    custom_debugger = CustomDummyDebugger(output_key=output_key)

    datapoint = {
        'inputs': torch.randn(2, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (2,), dtype=torch.int64),
        'outputs': torch.randn(2, 10, dtype=torch.float32),
        'meta_info': {'idx': [0]}
    }

    result = custom_debugger(datapoint, dummy_model)
    assert output_key in result
    assert len(result) == 1


@pytest.mark.parametrize("test_case,description", [
    (torch.tensor([[0.001]], dtype=torch.float32), "very_small_outputs"),
    (torch.randn(1, 1000, dtype=torch.float32) * 1000, "large_outputs"),
    (torch.tensor([[0.0]], dtype=torch.float32), "zero_outputs"),
])
def test_debugger_edge_cases(dummy_model, test_case, description):
    """Test debugger behavior with edge cases."""

    class EdgeCaseDummyDebugger(BaseDebugger):
        def __init__(self, output_key: str = "dummy_stats"):
            self.output_key = output_key

        def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
            outputs = datapoint['outputs']
            stats = {
                'mean': torch.mean(outputs).item(),
                'std': torch.std(outputs).item(),
                'min': torch.min(outputs).item(),
                'max': torch.max(outputs).item(),
            }
            return {self.output_key: stats}

    debugger = EdgeCaseDummyDebugger()

    datapoint = {
        'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (1,), dtype=torch.int64),
        'outputs': test_case,
        'meta_info': {'idx': [0]}
    }

    # Test cases that should work normally
    result = debugger(datapoint, dummy_model)
    stats = result["dummy_stats"]
    assert all(isinstance(v, float) for v in stats.values())


@pytest.mark.parametrize("invalid_input,expected_exception", [
    ("not_a_tensor", TypeError),
    (None, TypeError),
    ([], TypeError),
    ({}, TypeError),
])
def test_debugger_invalid_inputs(dummy_model, invalid_input, expected_exception):
    """Test debugger behavior with invalid inputs (Invalid Input Testing Pattern)."""

    class InvalidInputDummyDebugger(BaseDebugger):
        def __init__(self, output_key: str = "dummy_stats"):
            self.output_key = output_key

        def __call__(self, datapoint: Dict[str, Dict[str, Any]], model: torch.nn.Module) -> Dict[str, Any]:
            outputs = datapoint['outputs']
            stats = {
                'mean': torch.mean(outputs).item(),
                'std': torch.std(outputs).item(),
                'min': torch.min(outputs).item(),
                'max': torch.max(outputs).item(),
            }
            return {self.output_key: stats}

    debugger = InvalidInputDummyDebugger()

    datapoint = {
        'inputs': torch.randn(1, 3, 32, 32, dtype=torch.float32),
        'labels': torch.randint(0, 10, (1,), dtype=torch.int64),
        'outputs': invalid_input,
        'meta_info': {'idx': [0]}
    }

    # Test invalid inputs raise appropriate exceptions
    with pytest.raises(expected_exception):
        debugger(datapoint, dummy_model)
