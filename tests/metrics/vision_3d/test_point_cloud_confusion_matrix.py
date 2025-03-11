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


def test_imbalanced_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    num_points = 1000
    
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Create highly imbalanced data where class 0 dominates
    y_true = torch.zeros(num_points, device=device, dtype=torch.int64)
    y_true[900:950] = 1  # 5% class 1
    y_true[950:] = 2     # 5% class 2
    
    # Create predictions that always predict the majority class
    y_pred = torch.zeros(num_points, num_classes, device=device)
    y_pred[:, 0] = 1.0
    
    metric(y_pred, y_true)
    summary = metric.summarize()
    
    # Check that precision and recall are correct for the majority class
    assert torch.isclose(summary['class_precision'][0], torch.tensor(0.9, device=device)), \
        "Precision for majority class should be 0.9"
    assert torch.isclose(summary['class_recall'][0], torch.tensor(1.0, device=device)), \
        "Recall for majority class should be 1.0"
    
    # Check that minority classes have 0 recall
    assert torch.all(summary['class_recall'][1:] == 0), \
        "Recall for minority classes should be 0"


def test_save_functionality(tmp_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    num_points = 100
    
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Create some predictions
    y_pred = torch.randn(num_points, num_classes, device=device)
    y_true = torch.randint(0, num_classes, (num_points,), device=device)
    
    metric(y_pred, y_true)
    
    # Save results to a temporary file
    output_path = str(tmp_path / "results.json")
    summary = metric.summarize(output_path=output_path)
    
    # Check that file exists
    assert tmp_path.joinpath("results.json").exists(), "Results file should exist"


def test_zero_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    num_points = 100
    
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Create predictions where class 2 is never predicted
    y_pred = torch.zeros(num_points, num_classes, device=device)
    logits = torch.randn(num_points, 2, device=device)  # Only generate logits for classes 0 and 1
    y_pred[:, :2] = torch.nn.functional.softmax(logits, dim=1)  # Ensure valid probabilities
    y_true = torch.randint(0, num_classes, (num_points,), device=device)
    
    metric(y_pred, y_true)
    summary = metric.summarize()
    
    # For class 2 (never predicted):
    # - True positives should be 0 (never correctly predicted)
    # - False positives should be 0 (never predicted)
    assert summary['tp'][2] == 0, "True positives should be 0 for never-predicted class"
    assert summary['fp'][2] == 0, "False positives should be 0 for never-predicted class"
    
    # Precision should be 0 due to epsilon in denominator (0 / (0 + 0 + epsilon) = 0)
    assert summary['class_precision'][2] == 0, \
        "Precision for never-predicted class should be 0"
    
    # Recall should be 0 for the never-predicted class if there are ground truth instances
    if summary['tp'][2] + summary['fn'][2] > 0:  # If there are ground truth instances
        assert summary['class_recall'][2] == 0, \
            "Recall for never-predicted class should be 0 if there are ground truth instances"


def test_device_placement():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    num_classes = 2
    num_points = 100
    
    # Initialize metric (metric internal state should be on CPU)
    metric = PointCloudConfusionMatrix(num_classes=num_classes)
    
    # Test with CPU tensors
    y_pred_cpu = torch.randn(num_points, num_classes)
    y_true_cpu = torch.randint(0, num_classes, (num_points,))
    metric(y_pred_cpu, y_true_cpu)
    
    # Test with CUDA tensors
    y_pred_cuda = torch.randn(num_points, num_classes, device="cuda")
    y_true_cuda = torch.randint(0, num_classes, (num_points,), device="cuda")
    metric(y_pred_cuda, y_true_cuda)
    
    # Summary should work regardless of input device
    summary = metric.summarize()
    assert 'accuracy' in summary, "Summary should work with mixed device inputs"
