import pytest
import torch
from metrics.vision_3d.point_cloud_confusion_matrix import PointCloudConfusionMatrix


@pytest.mark.parametrize("num_classes, num_points", [
    (2, 1000),  # Binary classification
    (4, 500),   # Multi-class with fewer points
    (10, 2000), # Multi-class with more points
])
def test_point_cloud_confusion_matrix(num_classes, num_points):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize metric
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Create dummy predictions and targets
    y_pred = torch.randn(num_points, num_classes, device=device)
    y_true = torch.randint(0, num_classes, (num_points,), device=device)
    
    # Update metric
    metric(y_pred, y_true)
    
    # Get summary
    summary = metric.summarize()
    
    # Check that all expected metrics are present
    expected_keys = {
        'tp', 'tn', 'fp', 'fn',
        'class_accuracy', 'class_precision', 'class_recall', 'class_f1',
        'accuracy', 'mean_precision', 'mean_recall', 'mean_f1'
    }
    assert set(summary.keys()) == expected_keys
    
    # Check shapes
    assert summary['tp'].shape == (num_classes,)
    assert summary['tn'].shape == (num_classes,)
    assert summary['fp'].shape == (num_classes,)
    assert summary['fn'].shape == (num_classes,)
    assert summary['class_accuracy'].shape == (num_classes,)
    assert summary['class_precision'].shape == (num_classes,)
    assert summary['class_recall'].shape == (num_classes,)
    assert summary['class_f1'].shape == (num_classes,)
    assert summary['accuracy'].shape == ()
    assert summary['mean_precision'].shape == ()
    assert summary['mean_recall'].shape == ()
    assert summary['mean_f1'].shape == ()
    
    # Check value ranges
    for key in ['class_accuracy', 'class_precision', 'class_recall', 'class_f1',
                'accuracy', 'mean_precision', 'mean_recall', 'mean_f1']:
        assert torch.all((0 <= summary[key]) & (summary[key] <= 1)), f"Values out of range [0,1] for {key}"


def test_perfect_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    num_points = 100
    
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Create perfect predictions
    y_true = torch.randint(0, num_classes, (num_points,), device=device)
    y_pred = torch.zeros(num_points, num_classes, device=device)
    y_pred[torch.arange(num_points), y_true] = 1.0
    
    metric(y_pred, y_true)
    summary = metric.summarize()
    
    # For perfect predictions, accuracy should be 1
    assert torch.isclose(summary['accuracy'], torch.tensor(1.0)), "Perfect predictions should have accuracy 1"


def test_bincount_computation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    
    # Create a simple case with known outcome
    y_pred = torch.tensor([0, 0, 1, 1], device=device)
    y_true = torch.tensor([0, 1, 0, 1], device=device)
    
    # Expected confusion matrix:
    # [[1, 1],
    #  [1, 1]]
    expected_bincount = torch.tensor([[1, 1], [1, 1]], device=device)
    
    computed_bincount = PointCloudConfusionMatrix._get_bincount(y_pred, y_true, num_classes)
    assert torch.equal(computed_bincount, expected_bincount), "Bincount computation incorrect"


def test_multiple_updates():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Update with multiple batches
    for _ in range(3):
        y_pred = torch.randn(100, num_classes, device=device)
        y_true = torch.randint(0, num_classes, (100,), device=device)
        metric(y_pred, y_true)
    
    # Summary should work with multiple updates
    summary = metric.summarize()
    assert 'accuracy' in summary, "Summary should work after multiple updates"
