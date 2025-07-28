"""Tests for D3Feat criteria."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

import pytest
import torch

from criteria.vision_3d.point_cloud_registration.d3feat_criteria import (
    CircleLoss, ContrastiveLoss
)


def test_circle_loss_initialization():
    """Test CircleLoss initialization."""
    # Default initialization
    criterion = CircleLoss()
    assert criterion is not None
    assert hasattr(criterion, 'DIRECTIONS')
    assert criterion.DIRECTIONS['circle_loss'] == -1  # Lower is better
    
    # Custom initialization
    criterion = CircleLoss(
        log_scale=20.0,
        pos_margin=0.2,
        neg_margin=1.5,
        desc_loss_weight=2.0,
        det_loss_weight=0.5
    )
    assert criterion.desc_loss_weight == 2.0
    assert criterion.det_loss_weight == 0.5


def test_circle_loss_forward():
    """Test CircleLoss forward pass."""
    criterion = CircleLoss()
    
    # Create dummy predictions
    batch_size = 1
    num_points = 256
    feature_dim = 32
    num_corr = 20
    
    # Predictions
    descriptors = torch.randn(2 * num_points, feature_dim)
    # Normalize descriptors
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.sigmoid(torch.randn(2 * num_points, 1))
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    # Ground truth
    correspondences = torch.randint(0, num_points, (num_corr, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Forward pass
    losses = criterion(y_pred, y_true)
    
    # Check outputs
    assert 'circle_loss' in losses
    assert 'desc_loss' in losses
    assert 'det_loss' in losses
    assert 'accuracy' in losses
    
    # Check values are reasonable
    assert not torch.isnan(losses['circle_loss'])
    assert not torch.isinf(losses['circle_loss'])
    assert losses['accuracy'] >= 0 and losses['accuracy'] <= 100


def test_circle_loss_no_correspondences():
    """Test CircleLoss with no correspondences."""
    criterion = CircleLoss()
    
    # Create predictions
    descriptors = torch.randn(100, 32)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.sigmoid(torch.randn(100, 1))
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    # Empty correspondences
    y_true = {
        'correspondences': torch.zeros(0, 2, dtype=torch.long),
    }
    
    # Forward pass
    losses = criterion(y_pred, y_true)
    
    # Should return zero loss
    assert losses['circle_loss'].item() == 0.0
    assert losses['desc_loss'].item() == 0.0
    assert losses['det_loss'].item() == 0.0
    assert losses['accuracy'].item() == 0.0


def test_contrastive_loss_initialization():
    """Test ContrastiveLoss initialization."""
    # Default initialization
    criterion = ContrastiveLoss()
    assert criterion is not None
    assert hasattr(criterion, 'DIRECTIONS')
    assert criterion.DIRECTIONS['contrastive_loss'] == -1
    
    # Custom initialization
    criterion = ContrastiveLoss(
        pos_margin=0.05,
        neg_margin=1.2,
        metric='cosine',
        safe_radius=0.3
    )
    assert criterion.contrastive_loss.pos_margin == 0.05
    assert criterion.contrastive_loss.neg_margin == 1.2


def test_contrastive_loss_forward():
    """Test ContrastiveLoss forward pass."""
    criterion = ContrastiveLoss()
    
    # Create dummy data
    num_points = 128
    feature_dim = 32
    num_corr = 15
    
    descriptors = torch.randn(2 * num_points, feature_dim)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.sigmoid(torch.randn(2 * num_points, 1))
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    correspondences = torch.randint(0, num_points, (num_corr, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Forward pass
    losses = criterion(y_pred, y_true)
    
    # Check outputs
    assert 'contrastive_loss' in losses
    assert 'desc_loss' in losses
    assert 'det_loss' in losses
    assert 'accuracy' in losses
    
    # Verify reasonable values
    assert not torch.isnan(losses['contrastive_loss'])
    assert not torch.isinf(losses['contrastive_loss'])


def test_loss_gradient_flow():
    """Test gradient flow through losses."""
    criterion = CircleLoss()
    
    # Create data with gradients
    num_points = 64
    feature_dim = 32
    num_corr = 10
    
    descriptors = torch.randn(2 * num_points, feature_dim, requires_grad=True)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.sigmoid(torch.randn(2 * num_points, 1, requires_grad=True))
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    correspondences = torch.randint(0, num_points, (num_corr, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Forward and backward
    losses = criterion(y_pred, y_true)
    total_loss = losses['circle_loss']
    total_loss.backward()
    
    # Check gradients exist
    assert descriptors.grad is not None
    assert scores.grad is not None


def test_loss_with_different_metrics():
    """Test losses with different distance metrics."""
    # Test CircleLoss with cosine distance
    criterion_cosine = CircleLoss(dist_type='cosine')
    
    # Test ContrastiveLoss with euclidean distance
    criterion_euclidean = ContrastiveLoss(metric='euclidean')
    
    # Create test data
    descriptors = torch.randn(100, 32)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.sigmoid(torch.randn(100, 1))
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    correspondences = torch.randint(0, 50, (10, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Test both
    losses_cosine = criterion_cosine(y_pred, y_true)
    losses_euclidean = criterion_euclidean(y_pred, y_true)
    
    # Both should produce valid losses
    assert not torch.isnan(losses_cosine['circle_loss'])
    assert not torch.isnan(losses_euclidean['contrastive_loss'])


if __name__ == '__main__':
    # Run tests
    test_circle_loss_initialization()
    print("✓ CircleLoss initialization test passed")
    
    test_circle_loss_forward()
    print("✓ CircleLoss forward pass test passed")
    
    test_circle_loss_no_correspondences()
    print("✓ CircleLoss no correspondences test passed")
    
    test_contrastive_loss_initialization()
    print("✓ ContrastiveLoss initialization test passed")
    
    test_contrastive_loss_forward()
    print("✓ ContrastiveLoss forward pass test passed")
    
    test_loss_gradient_flow()
    print("✓ Loss gradient flow test passed")
    
    test_loss_with_different_metrics()
    print("✓ Loss with different metrics test passed")
    
    print("\nAll D3Feat criteria tests passed!")