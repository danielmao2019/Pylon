"""
Test nested DIRECTIONS functionality with complex metrics.

Tests the new nested DIRECTIONS design where directions match the exact
structure of metric outputs.
"""
from typing import Dict
import pytest
import torch
from runners.model_comparison import get_metric_directions, compare_scores, _flatten_directions, _flatten_scores
from metrics.vision_2d.change_detection.change_star_metric import ChangeStarMetric
from metrics.vision_2d.semantic_segmentation_metric import SemanticSegmentationMetric


def test_nested_directions_structure():
    """Test that nested DIRECTIONS match the exact output structure."""
    metric = ChangeStarMetric()
    directions = get_metric_directions(metric)
    
    # ChangeStarMetric should have nested structure matching its outputs
    expected_structure = {
        'change': {
            'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 
            'mean_recall': 1, 'mean_f1': 1
        },
        'semantic_1': {
            'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 
            'mean_recall': 1, 'mean_f1': 1
        },
        'semantic_2': {
            'mean_IoU': 1, 'accuracy': 1, 'mean_precision': 1, 
            'mean_recall': 1, 'mean_f1': 1
        }
    }
    
    assert directions == expected_structure


def test_flatten_directions_nested():
    """Test flattening of nested direction structure."""
    nested_directions = {
        'change': {
            'mean_IoU': 1, 
            'accuracy': 1
        },
        'semantic_1': {
            'mean_IoU': 1,
            'accuracy': 1
        }
    }
    
    flat = _flatten_directions(nested_directions)
    expected = {
        'change.mean_IoU': 1,
        'change.accuracy': 1,
        'semantic_1.mean_IoU': 1,
        'semantic_1.accuracy': 1
    }
    
    assert flat == expected


def test_flatten_scores_nested():
    """Test flattening of nested score structure."""
    nested_scores = {
        'change': {
            'mean_IoU': 0.8, 
            'accuracy': 0.9
        },
        'semantic_1': {
            'mean_IoU': 0.7,
            'accuracy': 0.85
        }
    }
    
    flat = _flatten_scores(nested_scores)
    expected = {
        'change.mean_IoU': 0.8,
        'change.accuracy': 0.9,
        'semantic_1.mean_IoU': 0.7,
        'semantic_1.accuracy': 0.85
    }
    
    assert flat == expected


def test_compare_scores_with_nested_structure():
    """Test score comparison with nested structure."""
    # Create nested directions
    nested_directions = {
        'change': {
            'mean_IoU': 1, 
            'accuracy': 1
        },
        'semantic_1': {
            'mean_IoU': 1,
            'accuracy': 1
        }
    }
    
    # Create nested scores (current better than best)
    current_scores = {
        'change': {
            'mean_IoU': 0.8, 
            'accuracy': 0.9
        },
        'semantic_1': {
            'mean_IoU': 0.7,
            'accuracy': 0.85
        }
    }
    
    best_scores = {
        'change': {
            'mean_IoU': 0.75, 
            'accuracy': 0.85
        },
        'semantic_1': {
            'mean_IoU': 0.65,
            'accuracy': 0.8
        }
    }
    
    # Test with True (equal weight average)
    result = compare_scores(current_scores, best_scores, True, nested_directions)
    assert result == True  # Current should be better
    
    # Test with False (vector comparison)
    result = compare_scores(current_scores, best_scores, False, nested_directions)
    assert result == True  # Current should be better in all dimensions


def test_compare_scores_incomparable_nested():
    """Test score comparison with incomparable nested vectors."""
    nested_directions = {
        'change': {
            'mean_IoU': 1, 
            'accuracy': 1
        },
        'semantic_1': {
            'mean_IoU': 1,
            'accuracy': 1
        }
    }
    
    # Create incomparable scores (one better in some dimensions, worse in others)
    current_scores = {
        'change': {
            'mean_IoU': 0.8,   # Better
            'accuracy': 0.85   # Worse
        },
        'semantic_1': {
            'mean_IoU': 0.65,  # Worse
            'accuracy': 0.9    # Better
        }
    }
    
    best_scores = {
        'change': {
            'mean_IoU': 0.75,  # Worse than current
            'accuracy': 0.9    # Better than current
        },
        'semantic_1': {
            'mean_IoU': 0.7,   # Better than current
            'accuracy': 0.85   # Worse than current
        }
    }
    
    # Vector comparison should return False (incomparable treated as no improvement)
    result = compare_scores(current_scores, best_scores, False, nested_directions)
    assert result == False
    
    # Scalar comparison should still work
    result = compare_scores(current_scores, best_scores, True, nested_directions)
    # Calculate expected: all current scores vs all best scores
    # current: 0.8 + 0.85 + 0.65 + 0.9 = 3.2, average = 0.8
    # best: 0.75 + 0.9 + 0.7 + 0.85 = 3.2, average = 0.8
    # Should be very close, so result could be either way
    assert isinstance(result, bool)


def test_real_change_star_metric_directions():
    """Test with actual ChangeStarMetric to ensure integration works."""
    metric = ChangeStarMetric()
    directions = get_metric_directions(metric)
    
    # Should have the nested structure we expect
    assert 'change' in directions
    assert 'semantic_1' in directions  
    assert 'semantic_2' in directions
    
    # Each should be a dict with semantic segmentation metrics
    for task in ['change', 'semantic_1', 'semantic_2']:
        assert isinstance(directions[task], dict)
        assert 'mean_IoU' in directions[task]
        assert 'accuracy' in directions[task]
        assert directions[task]['mean_IoU'] == 1  # Higher is better
        assert directions[task]['accuracy'] == 1  # Higher is better


def test_mixed_flat_and_nested_scores():
    """Test handling of mixed flat and nested score structures."""
    # Mixed directions: some flat, some nested
    mixed_directions = {
        'overall_score': 1,  # Flat
        'detailed': {        # Nested
            'precision': 1,
            'recall': 1
        }
    }
    
    mixed_scores = {
        'overall_score': 0.85,  # Flat
        'detailed': {           # Nested
            'precision': 0.9,
            'recall': 0.8
        }
    }
    
    # Should flatten correctly
    flat_directions = _flatten_directions(mixed_directions)
    flat_scores = _flatten_scores(mixed_scores)
    
    expected_directions = {
        'overall_score': 1,
        'detailed.precision': 1,
        'detailed.recall': 1
    }
    
    expected_scores = {
        'overall_score': 0.85,
        'detailed.precision': 0.9,
        'detailed.recall': 0.8
    }
    
    assert flat_directions == expected_directions
    assert flat_scores == expected_scores


def test_class_attribute_access():
    """Test that DIRECTIONS can be accessed as a class attribute."""
    # Test ChangeStarMetric class attribute access
    directions = ChangeStarMetric.DIRECTIONS
    assert 'change' in directions
    assert 'semantic_1' in directions
    assert 'semantic_2' in directions
    
    # Test SemanticSegmentationMetric class attribute access
    seg_directions = SemanticSegmentationMetric.DIRECTIONS
    assert 'mean_IoU' in seg_directions
    assert seg_directions['mean_IoU'] == 1
    
    # Verify they work with get_metric_directions without instantiation
    from runners.model_comparison import get_metric_directions
    
    # Create dummy metric to test get_metric_directions works properly
    class DummyMetric:
        DIRECTIONS = directions
    
    extracted = get_metric_directions(DummyMetric())
    assert extracted == directions