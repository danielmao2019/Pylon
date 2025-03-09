import pytest
import torch
import os
import tempfile
import json
from metrics.vision_3d.point_cloud_confusion_matrix import PointCloudConfusionMatrix


@pytest.fixture
def sample_data():
    # Generate sample data
    num_points_per_sample = 100
    num_classes = 4
    batch_size = 2
    total_points = num_points_per_sample * batch_size
    
    # Create logits for all points ([N, C] format)
    logits = torch.zeros(total_points, num_classes)
    
    # Create labels for all points ([N] format)
    labels = torch.zeros(total_points, dtype=torch.int64)
    
    # Create a batch indicator tensor for testing
    batch = torch.zeros(total_points, dtype=torch.int64)
    
    # Fill in data with a mix of correct and incorrect predictions
    for i in range(total_points):
        # Determine which sample this point belongs to
        sample_idx = i // num_points_per_sample
        batch[i] = sample_idx
        
        # Set up class labels and predictions
        true_class = i % num_classes
        pred_class = (i % 2 == 0) and true_class or (true_class + 1) % num_classes
        
        # Record the label
        labels[i] = true_class
        
        # Make the prediction confident
        logits[i, pred_class] = 10.0
    
    return {
        'logits': logits,
        'labels': labels,
        'batch': batch,
        'num_classes': num_classes,
        'num_points_per_sample': num_points_per_sample,
        'batch_size': batch_size
    }


def test_point_cloud_confusion_matrix_init():
    """Test that we can initialize the metric."""
    num_classes = 10
    metric = PointCloudConfusionMatrix(num_classes)
    assert metric.num_classes == num_classes
    assert len(metric.buffer) == 0


def test_point_cloud_confusion_matrix_compute_score(sample_data):
    """Test that we can compute scores with the metric."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Compute scores
    scores = metric._compute_score(sample_data['logits'], sample_data['labels'])
    
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
    assert torch.all(total == sample_data['labels'].size(0))


def test_point_cloud_confusion_matrix_call(sample_data):
    """Test that we can call the metric with various input formats."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Direct tensors
    scores1 = metric(sample_data['logits'], sample_data['labels'])
    assert isinstance(scores1, dict)
    assert 'tp' in scores1
    
    # Dictionary inputs
    scores2 = metric({'pred': sample_data['logits']}, {'true': sample_data['labels']})
    assert isinstance(scores2, dict)
    assert 'tp' in scores2
    
    # Buffer should contain the scores
    assert len(metric.buffer) == 2
    assert all(isinstance(x, dict) for x in metric.buffer)


def test_point_cloud_confusion_matrix_summarize(sample_data):
    """Test that we can summarize the results of the metric."""
    metric = PointCloudConfusionMatrix(sample_data['num_classes'])
    
    # Add some scores to the buffer
    for _ in range(3):
        metric(sample_data['logits'], sample_data['labels'])
    
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
    for _ in range(3):
        metric(sample_data['logits'], sample_data['labels'])
    
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


def test_point_cloud_confusion_matrix_perfect_score():
    """Test that perfect predictions give perfect scores."""
    num_points = 100
    num_classes = 3
    
    # Create perfect predictions
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


def test_point_cloud_confusion_matrix_per_sample_metrics():
    """Test that we can compute metrics separately for each sample in the batch."""
    num_points_per_sample = 50
    num_classes = 3
    batch_size = 2
    total_points = num_points_per_sample * batch_size
    
    # Create data with different accuracy for each sample
    labels = torch.zeros(total_points, dtype=torch.int64)
    logits = torch.zeros(total_points, num_classes)
    batch = torch.zeros(total_points, dtype=torch.int64)
    
    for i in range(total_points):
        # Determine which sample this point belongs to
        sample_idx = i // num_points_per_sample
        batch[i] = sample_idx
        
        # Set class label
        true_class = i % num_classes
        labels[i] = true_class
        
        # First sample: all correct predictions
        # Second sample: all incorrect predictions
        if sample_idx == 0:
            pred_class = true_class  # Correct prediction
        else:
            pred_class = (true_class + 1) % num_classes  # Incorrect prediction
        
        # Make the prediction confident
        logits[i, pred_class] = 10.0
    
    # Compute metrics on all points
    metric = PointCloudConfusionMatrix(num_classes)
    scores = metric._compute_score(logits, labels)
    
    # Compute metrics per sample
    sample_metrics = []
    for b in range(batch_size):
        mask = batch == b
        sample_logits = logits[mask]
        sample_labels = labels[mask]
        
        sample_metric = PointCloudConfusionMatrix(num_classes)
        sample_scores = sample_metric._compute_score(sample_logits, sample_labels)
        sample_metrics.append(sample_scores)
    
    # Check that per-sample metrics are different
    # Sample 0 should have perfect predictions (all TP, no FP or FN)
    # Sample 1 should have no correct predictions (no TP, all FP and FN)
    for c in range(num_classes):
        assert sample_metrics[0]['tp'][c] > 0
        assert sample_metrics[0]['fp'][c] == 0
        assert sample_metrics[0]['fn'][c] == 0
        
        assert sample_metrics[1]['tp'][c] == 0
        assert sample_metrics[1]['fp'][c] > 0 or sample_metrics[1]['fn'][c] > 0
    
    # The combined metrics should be somewhere in between
    # Check that TP is equal to the sum of the per-sample TPs
    assert torch.allclose(scores['tp'], sample_metrics[0]['tp'] + sample_metrics[1]['tp'])
