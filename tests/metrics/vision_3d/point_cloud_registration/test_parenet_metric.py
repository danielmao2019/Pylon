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


def test_parenet_metric_computation():
    """Test PARENet metric computation with proper datapoint format."""
    metric = build_from_config(metric_cfg)
    
    # Create properly formatted datapoint matching metric expectations
    dummy_datapoint = {
        'outputs': {
            'estimated_transform': torch.eye(4),
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'src_points': torch.randn(100, 3),
            # Required coarse-level outputs for PARENet evaluator
            'ref_points_c': torch.randn(64, 3),
            'src_points_c': torch.randn(64, 3),
            'gt_node_corr_indices': torch.randint(0, 64, (20, 2)),
            'gt_node_corr_overlaps': torch.rand(20),
            'ref_node_corr_indices': torch.randint(0, 64, (20,)),
            'src_node_corr_indices': torch.randint(0, 64, (20,)),
        },
        'labels': {
            'transform': torch.eye(4),
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 0,
        }
    }
    
    # Call metric to compute metrics - this should work without errors
    result = metric(dummy_datapoint)
    
    # Verify metric computation
    assert isinstance(result, dict), "Metric result must be a dictionary"
    assert len(result) > 0, "Metric result should contain computed metrics"
    
    # Check for expected metric keys
    expected_metrics = ['rotation_error', 'translation_error', 'inlier_ratio', 
                       'point_inlier_ratio', 'fine_precision', 'rmse', 'registration_recall']
    for metric_name in expected_metrics:
        assert metric_name in result, f"Missing expected metric: {metric_name}"
        assert isinstance(result[metric_name], torch.Tensor), f"Metric {metric_name} must be a tensor"


def test_parenet_metric_multiple_calls():
    """Test PARENet metric with multiple calls to verify consistency."""
    metric = build_from_config(metric_cfg)
    
    # Create two different datapoints
    dummy_datapoint1 = {
        'outputs': {
            'estimated_transform': torch.eye(4),
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'src_points': torch.randn(100, 3),
            # Required coarse-level outputs for PARENet evaluator
            'ref_points_c': torch.randn(64, 3),
            'src_points_c': torch.randn(64, 3),
            'gt_node_corr_indices': torch.randint(0, 64, (20, 2)),
            'gt_node_corr_overlaps': torch.rand(20),
            'ref_node_corr_indices': torch.randint(0, 64, (20,)),
            'src_node_corr_indices': torch.randint(0, 64, (20,)),
        },
        'labels': {
            'transform': torch.eye(4),
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 0,
        }
    }
    
    dummy_datapoint2 = {
        'outputs': {
            'estimated_transform': torch.eye(4) * 1.1,  # Slightly different transform
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'src_points': torch.randn(100, 3),
            # Required coarse-level outputs for PARENet evaluator
            'ref_points_c': torch.randn(64, 3),
            'src_points_c': torch.randn(64, 3),
            'gt_node_corr_indices': torch.randint(0, 64, (20, 2)),
            'gt_node_corr_overlaps': torch.rand(20),
            'ref_node_corr_indices': torch.randint(0, 64, (20,)),
            'src_node_corr_indices': torch.randint(0, 64, (20,)),
        },
        'labels': {
            'transform': torch.eye(4),
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 1,
        }
    }
    
    # Test multiple metric calls
    result1 = metric(dummy_datapoint1)
    result2 = metric(dummy_datapoint2)
    
    # Verify both results have expected structure
    for result in [result1, result2]:
        assert isinstance(result, dict), "Metric result must be a dictionary"
        expected_metrics = ['rotation_error', 'translation_error', 'inlier_ratio', 
                           'point_inlier_ratio', 'fine_precision', 'rmse', 'registration_recall']
        for metric_name in expected_metrics:
            assert metric_name in result, f"Missing metric: {metric_name}"
            assert isinstance(result[metric_name], torch.Tensor), f"Metric {metric_name} must be a tensor"


def test_parenet_metric_batch_dimensions():
    """Test PARENet metric handling of batch dimensions."""
    metric = build_from_config(metric_cfg)
    
    # Test with consistent dimensions matching model output
    dummy_datapoint = {
        'outputs': {
            'estimated_transform': torch.eye(4),  # Model outputs (4, 4) not (1, 4, 4)
            'ref_corr_points': torch.randn(50, 3),
            'src_corr_points': torch.randn(50, 3),
            'src_points': torch.randn(100, 3),
            # Required coarse-level outputs for PARENet evaluator
            'ref_points_c': torch.randn(64, 3),
            'src_points_c': torch.randn(64, 3),
            'gt_node_corr_indices': torch.randint(0, 64, (20, 2)),
            'gt_node_corr_overlaps': torch.rand(20),
            'ref_node_corr_indices': torch.randint(0, 64, (20,)),
            'src_node_corr_indices': torch.randint(0, 64, (20,)),
        },
        'labels': {
            'transform': torch.eye(4),  # Match model output format
            'src_points': torch.randn(100, 3),
        },
        'meta_info': {
            'idx': 0,
        }
    }
    
    # Should handle batch dimensions correctly
    result = metric(dummy_datapoint)
    
    # Verify result structure
    assert isinstance(result, dict), "Metric result must be a dictionary"
    expected_metrics = ['rotation_error', 'translation_error', 'inlier_ratio', 
                       'point_inlier_ratio', 'fine_precision', 'rmse', 'registration_recall']
    for metric_name in expected_metrics:
        assert metric_name in result, f"Missing metric: {metric_name}"
        assert isinstance(result[metric_name], torch.Tensor), f"Metric {metric_name} must be a tensor"


def test_parenet_metric_component_metrics():
    """Test PARENet metric component initialization."""
    metric = build_from_config(metric_cfg)
    
    # Check that component metrics are initialized
    assert hasattr(metric, 'isotropic_error')
    assert hasattr(metric, 'inlier_ratio')
    assert hasattr(metric, 'parenet_evaluator')
    
    print("✓ Component metrics properly initialized")
