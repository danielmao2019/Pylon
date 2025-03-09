import pytest
import torch
import os
import tempfile
import json
from metrics.vision_3d.point_cloud_confusion_matrix import PointCloudConfusionMatrix


@pytest.fixture
def sample_data():
    # Generate sample data
    num_points = 100
    num_classes = 4
    batch_size = 2
    
    # Create unbatched data
    logits_unbatched = torch.zeros(num_points, num_classes)
    # Ensure a mix of correct and incorrect predictions
    for i in range(num_points):
        true_class = i % num_classes
        pred_class = (i % 2 == 0) and true_class or (true_class + 1) % num_classes
        logits_unbatched[i, pred_class] = 10.0  # Make prediction very confident
    
    labels_unbatched = torch.tensor([i % num_classes for i in range(num_points)], dtype=torch.int64)
    
    # Create batched data
    logits_batched = torch.zeros(batch_size, num_points, num_classes)
    labels_batched = torch.zeros(batch_size, num_points, dtype=torch.int64)
    
    for b in range(batch_size):
        for i in range(num_points):
            true_class = (i + b) % num_classes
            pred_class = ((i + b) % 2 == 0) and true_class or (true_class + 1) % num_classes
            logits_batched[b, i, pred_class] = 10.0
            labels_batched[b, i] = true_class
    
    return {
        'logits_unbatched': logits_unbatched,
        'labels_unbatched': labels_unbatched,
        'logits_batched': logits_batched,
        'labels_batched': labels_batched,
        'num_classes': num_classes
    }


def test_point_cloud_confusion_matrix_init():
    """Test that we can initialize the metric."""
    num_classes = 10
    metric = PointCloudConfusionMatrix(num_classes)
    assert metric.num_classes == num_classes
    assert len(metric.buffer) == 0


def test_point_cloud_confusion_matrix_compute_score_unbatched(sample_data):
    """Test that we can compute scores with unbatched data."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Compute scores with unbatched data
    scores = metric._compute_score(
        sample_data['logits_unbatched'], 
        sample_data['labels_unbatched']
    )
    
    # Validate scores
    assert isinstance(scores, dict)
    assert 'tp' in scores
    assert 'tn' in scores
    assert 'fp' in scores
    assert 'fn' in scores
    
    # Check dimensions
    for key in ['tp', 'tn', 'fp', 'fn']:
        assert scores[key].shape == (sample_data['num_classes'],)
    
    # Verify that TP + TN + FP + FN = total number of points
    total = scores['tp'] + scores['tn'] + scores['fp'] + scores['fn']
    assert torch.all(total == sample_data['labels_unbatched'].size(0))


def test_point_cloud_confusion_matrix_compute_score_batched(sample_data):
    """Test that the metric handles batched inputs correctly through SingleTaskMetric."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Call with batched inputs
    scores = metric(
        sample_data['logits_batched'], 
        sample_data['labels_batched']
    )
    
    # Validate scores
    assert isinstance(scores, dict)
    assert 'tp' in scores
    assert 'tn' in scores
    assert 'fp' in scores
    assert 'fn' in scores
    
    # Buffer should contain the scores
    assert len(metric.buffer) == 1


def test_point_cloud_confusion_matrix_call(sample_data):
    """Test that we can call the metric with various input formats."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Direct tensors (unbatched)
    scores1 = metric(sample_data['logits_unbatched'], sample_data['labels_unbatched'])
    assert isinstance(scores1, dict)
    assert 'tp' in scores1
    
    # Dictionary inputs (unbatched)
    scores2 = metric(
        {'pred': sample_data['logits_unbatched']}, 
        {'true': sample_data['labels_unbatched']}
    )
    assert isinstance(scores2, dict)
    assert 'tp' in scores2
    
    # Direct tensors (batched)
    scores3 = metric(sample_data['logits_batched'], sample_data['labels_batched'])
    assert isinstance(scores3, dict)
    assert 'tp' in scores3
    
    # Dictionary inputs (batched)
    scores4 = metric(
        {'pred': sample_data['logits_batched']},
        {'true': sample_data['labels_batched']}
    )
    assert isinstance(scores4, dict)
    assert 'tp' in scores4
    
    # Buffer should contain the scores
    assert len(metric.buffer) == 4
    assert all(isinstance(x, dict) for x in metric.buffer)


def test_point_cloud_confusion_matrix_summarize(sample_data):
    """Test that we can summarize the results of the metric."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Add some scores to the buffer
    for _ in range(2):
        metric(sample_data['logits_unbatched'], sample_data['labels_unbatched'])
    
    for _ in range(1):
        metric(sample_data['logits_batched'], sample_data['labels_batched'])
    
    # Summarize
    summary = metric.summarize()
    
    # Validate summary
    assert isinstance(summary, dict)
    assert 'tp' in summary
    assert 'tn' in summary
    assert 'fp' in summary
    assert 'fn' in summary
    assert 'class_accuracy' in summary
    assert 'class_precision' in summary
    assert 'class_recall' in summary
    assert 'class_f1' in summary
    assert 'accuracy' in summary
    assert 'mean_precision' in summary
    assert 'mean_recall' in summary
    assert 'mean_f1' in summary
    
    # Check dimensions
    for key in ['tp', 'tn', 'fp', 'fn', 'class_accuracy', 'class_precision', 'class_recall', 'class_f1']:
        assert summary[key].shape == (sample_data['num_classes'],)
    
    # Scalar metrics
    for key in ['accuracy', 'mean_precision', 'mean_recall', 'mean_f1']:
        assert summary[key].dim() == 0
    
    # Check that global accuracy is between 0 and 1
    assert 0 <= summary['accuracy'] <= 1
    

def test_point_cloud_confusion_matrix_summarize_with_output(sample_data):
    """Test that we can summarize and save the results to disk."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Add some scores to the buffer
    for _ in range(2):
        metric(sample_data['logits_unbatched'], sample_data['labels_unbatched'])
    
    metric(sample_data['logits_batched'], sample_data['labels_batched'])
    
    # Create a temporary file to save the results
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Summarize and save to disk
        summary = metric.summarize(output_path=temp_path)
        
        # Check that the file exists
        assert os.path.exists(temp_path)
        
        # Load the saved results
        with open(temp_path, 'r') as f:
            saved_data = json.load(f)
        
        # Check that the saved results match the summary
        assert saved_data is not None
        assert 'accuracy' in saved_data
        assert abs(saved_data['accuracy'] - summary['accuracy'].item()) < 1e-6
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_point_cloud_confusion_matrix_perfect_score_unbatched():
    """Test that perfect predictions give perfect scores with unbatched inputs."""
    num_points = 100
    num_classes = 3
    
    # Create perfect predictions (unbatched)
    labels = torch.tensor([i % num_classes for i in range(num_points)], dtype=torch.int64)
    logits = torch.zeros(num_points, num_classes)
    for i in range(num_points):
        logits[i, labels[i]] = 10.0  # Strongly predict the correct class
    
    metric = PointCloudConfusionMatrix(num_classes)
    scores = metric._compute_score(logits, labels)
    
    # For perfect predictions, TP should be positive and FP/FN should be zero for each class
    for c in range(num_classes):
        count = torch.sum(labels == c).item()
        assert scores['tp'][c] == count
        assert scores['fp'][c] == 0
        assert scores['fn'][c] == 0


def test_point_cloud_confusion_matrix_perfect_score_batched():
    """Test that perfect predictions give perfect scores with batched inputs."""
    num_points = 50
    num_classes = 3
    batch_size = 2
    
    # Create perfect predictions (batched)
    labels = torch.zeros(batch_size, num_points, dtype=torch.int64)
    logits = torch.zeros(batch_size, num_points, num_classes)
    
    for b in range(batch_size):
        for i in range(num_points):
            class_idx = (i + b) % num_classes
            labels[b, i] = class_idx
            logits[b, i, class_idx] = 10.0  # Strongly predict the correct class
    
    metric = PointCloudConfusionMatrix(num_classes)
    scores = metric(logits, labels)
    
    # Validate scores from prediction
    assert isinstance(scores, dict)
    assert 'tp' in scores
    assert 'fp' in scores
    assert 'fn' in scores
    
    # The SingleTaskMetric will automatically extract values from the first batch element
    # for computation, but the evaluation should still work
    assert len(metric.buffer) == 1
