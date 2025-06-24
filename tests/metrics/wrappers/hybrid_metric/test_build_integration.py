import pytest
import torch
from metrics.base_metric import BaseMetric
from metrics.wrappers.hybrid_metric import HybridMetric
from utils.builders import build_from_config


def test_build_from_config_integration(dummy_metric, another_dummy_metric):
    """Test that HybridMetric works correctly with build_from_config."""
    # Test building HybridMetric from config
    hybrid_config = {
        'class': HybridMetric,
        'args': {
            'metrics_cfg': [
                {
                    'class': dummy_metric.__class__,
                    'args': {'metric_name': 'config_metric1'}
                },
                {
                    'class': another_dummy_metric.__class__,
                    'args': {'metric_name': 'config_metric2'}
                }
            ]
        }
    }

    # Build metric from config
    hybrid_metric = build_from_config(hybrid_config)

    # Verify it was built correctly
    assert isinstance(hybrid_metric, HybridMetric)
    assert len(hybrid_metric.metrics) == 2
    assert hybrid_metric.use_buffer is True

    # Test that it works functionally
    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    scores = hybrid_metric(y_pred=sample_input, y_true=sample_target)
    assert isinstance(scores, dict)
    assert 'config_metric1' in scores
    assert 'config_metric2' in scores


def test_build_from_config_with_buffer_disabled(dummy_metric):
    """Test building HybridMetric with disabled buffer from config."""
    hybrid_config = {
        'class': HybridMetric,
        'args': {
            'use_buffer': False,
            'metrics_cfg': [
                {
                    'class': dummy_metric.__class__,
                    'args': {'metric_name': 'no_buffer_metric'}
                }
            ]
        }
    }

    hybrid_metric = build_from_config(hybrid_config)

    assert isinstance(hybrid_metric, HybridMetric)
    assert hybrid_metric.use_buffer is False
    assert not hasattr(hybrid_metric, 'buffer')


def test_nested_config_building(dummy_metric, another_dummy_metric):
    """Test that nested configs are built correctly."""
    # Create a more complex nested config
    complex_config = {
        'class': HybridMetric,
        'args': {
            'metrics_cfg': [
                {
                    'class': dummy_metric.__class__,
                    'args': {
                        'metric_name': 'nested_metric1',
                        'use_buffer': True  # This should be overridden
                    }
                },
                {
                    'class': another_dummy_metric.__class__,
                    'args': {
                        'metric_name': 'nested_metric2',
                        'use_buffer': True  # This should be overridden
                    }
                }
            ]
        }
    }

    hybrid_metric = build_from_config(complex_config)

    # Verify nested components were built correctly with overridden buffer settings
    assert len(hybrid_metric.metrics) == 2
    for component in hybrid_metric.metrics:
        assert component.use_buffer is False
        assert not hasattr(component, 'buffer')


def test_config_parameter_merging(dummy_metric):
    """Test that config parameters are properly merged during building."""
    base_config = {
        'class': HybridMetric,
        'args': {
            'metrics_cfg': [
                {
                    'class': dummy_metric.__class__,
                    'args': {'metric_name': 'merge_test'}
                }
            ]
        }
    }

    # Test building with additional kwargs
    hybrid_metric = build_from_config(base_config, use_buffer=False)

    assert hybrid_metric.use_buffer is False
    assert not hasattr(hybrid_metric, 'buffer')


def test_recursive_building_preservation(dummy_metric):
    """Test that recursive building preserves object references correctly."""
    # Create config with shared references
    shared_metric_config = {
        'class': dummy_metric.__class__,
        'args': {'metric_name': 'shared'}
    }

    # Use the same config reference in multiple places
    hybrid_config = {
        'class': HybridMetric,
        'args': {
            'metrics_cfg': [shared_metric_config, shared_metric_config]
        }
    }

    # This should create two separate instances, not share the same instance
    hybrid_metric = build_from_config(hybrid_config)

    assert len(hybrid_metric.metrics) == 2
    # They should be different instances even though built from same config
    assert hybrid_metric.metrics[0] is not hybrid_metric.metrics[1]
    # But should have same class and attributes
    assert type(hybrid_metric.metrics[0]) == type(hybrid_metric.metrics[1])


def test_error_handling_in_build_process(dummy_metric):
    """Test error handling during the build process."""
    # Test with malformed config - create a class that requires an argument to force an error
    class RequiredArgMetric(BaseMetric):
        def __init__(self, required_arg, use_buffer=True):
            super().__init__(use_buffer=use_buffer)
            self.required_arg = required_arg
            
    malformed_config = {
        'class': HybridMetric,
        'args': {
            'metrics_cfg': [
                {
                    'class': RequiredArgMetric,
                    'args': {}  # Missing required_arg
                }
            ]
        }
    }

    with pytest.raises(Exception):  # Should fail during component building
        build_from_config(malformed_config)

    # Test with completely invalid config structure
    invalid_config = {
        'invalid_key': 'invalid_value'
    }

    # This should return the config as-is since it doesn't match expected structure
    result = build_from_config(invalid_config)
    assert result == invalid_config


def test_build_from_config_type_validation():
    """Test that build_from_config validates types correctly."""
    # Test with non-dict config
    non_dict_config = "not_a_dict"
    result = build_from_config(non_dict_config)
    assert result == non_dict_config

    # Test with list config
    list_config = [1, 2, 3]
    result = build_from_config(list_config)
    assert result == [1, 2, 3]  # Should process each element
