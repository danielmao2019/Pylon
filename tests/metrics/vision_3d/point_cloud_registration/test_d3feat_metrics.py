"""Tests for D3Feat metrics."""


import pytest
import torch
import numpy as np

from metrics.vision_3d.point_cloud_registration.d3feat_metrics import (
    D3FeatAccuracyMetric, D3FeatIoUMetric, D3FeatDescriptorMetric
)


def test_d3feat_accuracy_metric():
    """Test D3FeatAccuracyMetric."""
    metric = D3FeatAccuracyMetric()

    # Create dummy predictions and labels
    batch_size = 32
    num_classes = 5

    y_pred = torch.randn(batch_size, num_classes)
    y_true = torch.randint(0, num_classes, (batch_size,))

    # Create datapoint in expected format
    datapoint = {
        'outputs': {'predictions': y_pred},
        'labels': {'targets': y_true},
        'meta_info': {'idx': 0}
    }

    # Compute score
    scores = metric(datapoint)

    # Check outputs
    assert 'accuracy' in scores
    assert isinstance(scores['accuracy'], torch.Tensor)
    assert 0 <= scores['accuracy'].item() <= 100  # Percentage

    # Test DIRECTIONS
    assert hasattr(metric, 'DIRECTIONS')
    assert metric.DIRECTIONS['accuracy'] == 1  # Higher is better


def test_d3feat_iou_metric():
    """Test D3FeatIoUMetric."""
    num_classes = 3
    metric = D3FeatIoUMetric(num_classes=num_classes)

    # Create dummy predictions and labels
    batch_size = 24

    y_pred = torch.randn(batch_size, num_classes)
    y_true = torch.randint(0, num_classes, (batch_size,))

    # Create datapoint in expected format
    datapoint = {
        'outputs': {'predictions': y_pred},
        'labels': {'targets': y_true},
        'meta_info': {'idx': 0}
    }

    # Compute score
    scores = metric(datapoint)

    # Check outputs
    assert 'iou' in scores
    assert isinstance(scores['iou'], torch.Tensor)

    # Should have per-class IoU scores
    for i in range(num_classes):
        assert f'iou_class_{i}' in scores

    # Test DIRECTIONS
    assert hasattr(metric, 'DIRECTIONS')
    assert metric.DIRECTIONS['iou'] == 1  # Higher is better


def test_d3feat_descriptor_metric():
    """Test D3FeatDescriptorMetric."""
    metric = D3FeatDescriptorMetric(distance_threshold=0.1)

    # Create dummy descriptor predictions
    num_points = 128
    feature_dim = 32
    num_correspondences = 20

    # Simulated concatenated descriptors [src; tgt]
    descriptors = torch.randn(num_points, feature_dim)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

    # Simulated correspondences
    correspondences = torch.randint(0, num_points//2, (num_correspondences, 2))

    y_pred = {
        'descriptors': descriptors,
        'scores': torch.sigmoid(torch.randn(num_points, 1))
    }

    y_true = {
        'correspondences': correspondences
    }

    # Create datapoint in expected format
    datapoint = {
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': 0}
    }

    # Compute scores
    scores = metric(datapoint)

    # Check outputs
    assert 'desc_matching_accuracy' in scores
    assert 'feature_match_recall' in scores
    assert 'desc_distance' in scores

    # Check types and ranges
    assert isinstance(scores['desc_matching_accuracy'], torch.Tensor)
    assert isinstance(scores['feature_match_recall'], torch.Tensor)
    assert isinstance(scores['desc_distance'], torch.Tensor)

    assert 0 <= scores['desc_matching_accuracy'].item() <= 1
    assert 0 <= scores['feature_match_recall'].item() <= 1
    assert scores['desc_distance'].item() >= 0

    # Test DIRECTIONS
    assert hasattr(metric, 'DIRECTIONS')
    assert metric.DIRECTIONS['desc_matching_accuracy'] == 1  # Higher is better
    assert metric.DIRECTIONS['feature_match_recall'] == 1    # Higher is better
    assert metric.DIRECTIONS['desc_distance'] == -1         # Lower is better


def test_d3feat_descriptor_metric_empty_correspondences():
    """Test D3FeatDescriptorMetric with empty correspondences."""
    metric = D3FeatDescriptorMetric()

    # Create dummy data with no correspondences
    num_points = 64
    feature_dim = 32

    descriptors = torch.randn(num_points, feature_dim)
    correspondences = torch.zeros(0, 2, dtype=torch.long)  # Empty correspondences

    y_pred = {
        'descriptors': descriptors,
        'scores': torch.sigmoid(torch.randn(num_points, 1))
    }

    y_true = {
        'correspondences': correspondences
    }

    # Create datapoint in expected format
    datapoint = {
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': 0}
    }

    # Should handle empty correspondences gracefully
    scores = metric(datapoint)

    assert scores['desc_matching_accuracy'].item() == 0.0
    assert scores['feature_match_recall'].item() == 0.0
    assert scores['desc_distance'].item() == float('inf')


def test_d3feat_metrics_gradient_flow():
    """Test that metrics don't break gradient flow."""
    metric = D3FeatDescriptorMetric()

    # Create data with gradients
    num_points = 64
    feature_dim = 32
    num_correspondences = 10

    descriptors = torch.randn(num_points, feature_dim, requires_grad=True)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    descriptors.retain_grad()  # Retain gradient for non-leaf tensor
    correspondences = torch.randint(0, num_points//2, (num_correspondences, 2))

    y_pred = {
        'descriptors': descriptors,
        'scores': torch.sigmoid(torch.randn(num_points, 1))
    }

    y_true = {
        'correspondences': correspondences
    }

    # Create datapoint in expected format
    datapoint = {
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': 0}
    }

    # Compute scores
    scores = metric(datapoint)

    # Create dummy loss and backpropagate
    loss = scores['desc_distance']
    loss.backward()

    # Check gradients exist
    assert descriptors.grad is not None
    assert not torch.isnan(descriptors.grad).any()


def test_d3feat_metrics_device_consistency():
    """Test that metrics work correctly with different devices."""
    metric = D3FeatDescriptorMetric()

    # Test on CPU
    num_points = 32
    feature_dim = 16
    num_correspondences = 5

    descriptors = torch.randn(num_points, feature_dim)
    correspondences = torch.randint(0, num_points//2, (num_correspondences, 2))

    y_pred = {'descriptors': descriptors, 'scores': torch.randn(num_points, 1)}
    y_true = {'correspondences': correspondences}

    # Create datapoint in expected format
    datapoint = {
        'outputs': y_pred,
        'labels': y_true,
        'meta_info': {'idx': 0}
    }

    scores_cpu = metric(datapoint)

    # All outputs should be on the same device as inputs
    for key, value in scores_cpu.items():
        assert value.device == descriptors.device


if __name__ == '__main__':
    # Run tests
    test_d3feat_accuracy_metric()
    print("✓ D3FeatAccuracyMetric test passed")

    test_d3feat_iou_metric()
    print("✓ D3FeatIoUMetric test passed")

    test_d3feat_descriptor_metric()
    print("✓ D3FeatDescriptorMetric test passed")

    test_d3feat_descriptor_metric_empty_correspondences()
    print("✓ D3FeatDescriptorMetric empty correspondences test passed")

    test_d3feat_metrics_gradient_flow()
    print("✓ D3Feat metrics gradient flow test passed")

    test_d3feat_metrics_device_consistency()
    print("✓ D3Feat metrics device consistency test passed")

    print("\nAll D3Feat metrics tests passed!")