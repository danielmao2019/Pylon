"""
Unit tests for PARENet metric integration.

Tests PARENet evaluation metrics instantiation and computation.
"""

import pytest
import torch
from utils.builders import build_from_config
from configs.common.metrics.point_cloud_registration.parenet_metric_cfg import metric_cfg


def test_parenet_metric_instantiation():
    """Test that PARENet metric can be instantiated from config."""
    metric = build_from_config(metric_cfg)
    assert metric is not None
    assert hasattr(metric, 'DIRECTIONS')
    
    # Check DIRECTIONS is properly set
    expected_directions = ['rotation_error', 'translation_error', 'inlier_ratio', 
                          'point_inlier_ratio', 'fine_precision', 'rmse', 'registration_recall']
    for direction_key in expected_directions:
        assert direction_key in metric.DIRECTIONS
        assert metric.DIRECTIONS[direction_key] in [1, -1]


def test_parenet_metric_directions():
    """Test PARENet metric DIRECTIONS attribute."""
    metric = build_from_config(metric_cfg)
    
    # Test expected direction values
    assert metric.DIRECTIONS['rotation_error'] == -1  # Lower is better
    assert metric.DIRECTIONS['translation_error'] == -1  # Lower is better
    assert metric.DIRECTIONS['inlier_ratio'] == 1  # Higher is better
    assert metric.DIRECTIONS['point_inlier_ratio'] == 1  # Higher is better
    assert metric.DIRECTIONS['fine_precision'] == 1  # Higher is better
    assert metric.DIRECTIONS['rmse'] == -1  # Lower is better
    assert metric.DIRECTIONS['registration_recall'] == 1  # Higher is better


def test_parenet_metric_add_to_buffer():
    """Test PARENet metric buffer operations with proper datapoint format."""
    metric = build_from_config(metric_cfg)
    
    # Create properly formatted datapoint matching metric expectations
    dummy_datapoint = {
        'outputs': {
            'estimated_transform': torch.eye(4),
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'coarse_precision': torch.tensor(0.8),
            'fine_precision': torch.tensor(0.9),
            'rmse': torch.tensor(0.1),
            'registration_recall': torch.tensor(1.0),
        },
        'labels': {
            'transform': torch.eye(4),
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 0,
        }
    }
    
    # Add to buffer should work without errors
    metric.add_to_buffer(dummy_datapoint)
    
    # Verify buffer contains data
    assert len(metric.get_buffer()) > 0, "Buffer should contain added datapoint"


def test_parenet_metric_summarize():
    """Test PARENet metric summarization with proper data."""
    metric = build_from_config(metric_cfg)
    
    # Add a datapoint first
    dummy_datapoint = {
        'outputs': {
            'estimated_transform': torch.eye(4),
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'coarse_precision': torch.tensor(0.8),
            'fine_precision': torch.tensor(0.9),
            'rmse': torch.tensor(0.1),
            'registration_recall': torch.tensor(1.0),
        },
        'labels': {
            'transform': torch.eye(4),
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 0,
        }
    }
    
    metric.add_to_buffer(dummy_datapoint)
    
    # Test summarize with data
    scores = metric.summarize()
    
    # Verify summarize output structure
    assert isinstance(scores, dict), "Scores must be a dictionary"
    assert 'per_datapoint' in scores, "Scores must contain 'per_datapoint'"
    assert 'aggregated' in scores, "Scores must contain 'aggregated'"
    
    # Check that aggregated scores match expected metrics
    expected_metrics = ['rotation_error', 'translation_error', 'inlier_ratio', 
                       'point_inlier_ratio', 'fine_precision', 'rmse', 'registration_recall']
    
    for metric_name in expected_metrics:
        assert metric_name in scores['aggregated'], f"Missing metric: {metric_name}"
        assert isinstance(scores['aggregated'][metric_name], (int, float, torch.Tensor)), f"Invalid metric type for {metric_name}"


def test_parenet_metric_reset():
    """Test PARENet metric reset functionality."""
    metric = build_from_config(metric_cfg)
    
    # Add some data first
    dummy_datapoint = {
        'outputs': {
            'estimated_transform': torch.eye(4),
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'coarse_precision': torch.tensor(0.8),
            'fine_precision': torch.tensor(0.9),
            'rmse': torch.tensor(0.1),
            'registration_recall': torch.tensor(1.0),
        },
        'labels': {
            'transform': torch.eye(4),
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 0,
        }
    }
    
    metric.add_to_buffer(dummy_datapoint)
    assert len(metric.get_buffer()) > 0, "Buffer should contain data before reset"
    
    # Reset should clear the buffer
    metric.reset()
    assert len(metric.get_buffer()) == 0, "Buffer should be empty after reset"


def test_parenet_metric_component_metrics():
    """Test PARENet metric component initialization."""
    metric = build_from_config(metric_cfg)
    
    # Check that component metrics are initialized
    assert hasattr(metric, 'isotropic_error')
    assert hasattr(metric, 'inlier_ratio')
    assert hasattr(metric, 'parenet_evaluator')
    
    print("✓ Component metrics properly initialized")
