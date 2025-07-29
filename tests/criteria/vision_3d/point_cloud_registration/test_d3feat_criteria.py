"""Tests for D3Feat criteria."""

import pytest
import torch

from criteria.vision_3d.point_cloud_registration.d3feat_criteria.d3feat_criterion import D3FeatCriterion


def test_circle_loss_initialization():
    """Test CircleLoss initialization."""
    # Default initialization
    criterion = D3FeatCriterion(loss_type='circle')
    assert criterion is not None
    assert hasattr(criterion, 'DIRECTIONS')
    assert criterion.DIRECTIONS['circle_loss'] == -1  # Lower is better
    
    # Custom initialization
    criterion = D3FeatCriterion(
        loss_type='circle',
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
    criterion = D3FeatCriterion(loss_type='circle')
    
    # Create dummy predictions
    batch_size = 1
    num_points = 256
    feature_dim = 32
    num_corr = 20
    
    # Predictions
    descriptors = torch.randn(2 * num_points, feature_dim)
    # Normalize descriptors
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.abs(torch.randn(2 * num_points, 1))  # Positive scores like D3Feat detection_scores
    
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
    loss = criterion(y_pred, y_true)
    
    # Check outputs - should be a single tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1  # Single scalar tensor
    
    # Check values are reasonable
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_circle_loss_no_correspondences():
    """Test CircleLoss with no correspondences."""
    criterion = D3FeatCriterion(loss_type='circle')
    
    # Create predictions
    descriptors = torch.randn(100, 32)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.abs(torch.randn(100, 1))  # Positive scores like D3Feat detection_scores
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    # Empty correspondences
    y_true = {
        'correspondences': torch.zeros(0, 2, dtype=torch.long),
    }
    
    # Forward pass
    loss = criterion(y_pred, y_true)
    
    # Should return zero loss
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == 0.0


def test_contrastive_loss_initialization():
    """Test ContrastiveLoss initialization."""
    # Default initialization
    criterion = D3FeatCriterion(loss_type='contrastive')
    assert criterion is not None
    assert hasattr(criterion, 'DIRECTIONS')
    assert criterion.DIRECTIONS['contrastive_loss'] == -1
    
    # Custom initialization
    criterion = D3FeatCriterion(
        loss_type='contrastive',
        pos_margin=0.05,
        neg_margin=1.2,
        metric='cosine',
        safe_radius=0.3
    )
    assert criterion.descriptor_loss.pos_margin == 0.05
    assert criterion.descriptor_loss.neg_margin == 1.2


def test_contrastive_loss_forward():
    """Test ContrastiveLoss forward pass."""
    criterion = D3FeatCriterion(loss_type='contrastive')
    
    # Create dummy data
    num_points = 128
    feature_dim = 32
    num_corr = 15
    
    descriptors = torch.randn(2 * num_points, feature_dim)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.abs(torch.randn(2 * num_points, 1))  # Positive scores like D3Feat detection_scores
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    correspondences = torch.randint(0, num_points, (num_corr, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Forward pass
    loss = criterion(y_pred, y_true)
    
    # Check outputs - should be a single tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1  # Single scalar tensor
    
    # Verify reasonable values
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_loss_gradient_flow():
    """Test gradient flow through losses."""
    criterion = D3FeatCriterion(loss_type='circle')
    
    # Create data with gradients
    num_points = 64
    feature_dim = 32
    num_corr = 10
    
    descriptors_raw = torch.randn(2 * num_points, feature_dim, requires_grad=True)
    descriptors = torch.nn.functional.normalize(descriptors_raw, p=2, dim=1)
    scores_raw = torch.randn(2 * num_points, 1, requires_grad=True)
    scores = torch.abs(scores_raw)  # Positive scores like D3Feat detection_scores
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    correspondences = torch.randint(0, num_points, (num_corr, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Forward and backward
    loss = criterion(y_pred, y_true)
    loss.backward()
    
    # Check gradients exist (check the original tensors since normalize/sigmoid create non-leaf)
    assert descriptors_raw.grad is not None
    assert scores_raw.grad is not None


def test_loss_with_different_metrics():
    """Test losses with different distance metrics."""
    # Test D3FeatCriterion with CircleLoss using cosine distance
    criterion_cosine = D3FeatCriterion(loss_type='circle', dist_type='cosine')
    
    # Test D3FeatCriterion with ContrastiveLoss using euclidean distance
    criterion_euclidean = D3FeatCriterion(loss_type='contrastive', metric='euclidean')
    
    # Create test data
    descriptors = torch.randn(100, 32)
    descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
    scores = torch.abs(torch.randn(100, 1))  # Positive scores like D3Feat detection_scores
    
    y_pred = {
        'descriptors': descriptors,
        'scores': scores,
    }
    
    correspondences = torch.randint(0, 50, (10, 2), dtype=torch.long)
    y_true = {
        'correspondences': correspondences,
    }
    
    # Test both
    loss_cosine = criterion_cosine(y_pred, y_true)
    loss_euclidean = criterion_euclidean(y_pred, y_true)
    
    # Both should produce valid losses
    assert not torch.isnan(loss_cosine)
    assert not torch.isnan(loss_euclidean)
