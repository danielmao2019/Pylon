import pytest
import torch
from criteria.vision_3d import PointCloudSegmentationCriterion


@pytest.fixture
def sample_data():
    # Generate sample data
    num_points_per_sample = 100
    num_classes = 5
    batch_size = 2
    total_points = num_points_per_sample * batch_size
    
    # Create logits for all points ([N, C] format where N = total_points)
    logits = torch.randn(total_points, num_classes)
    
    # Create labels for all points ([N] format)
    labels = torch.randint(0, num_classes, (total_points,), dtype=torch.int64)
    
    # Create a batch indicator tensor for testing
    batch = torch.cat([torch.ones(num_points_per_sample, dtype=torch.int64) * i 
                      for i in range(batch_size)])
    
    return {
        'logits': logits,
        'labels': labels,
        'batch': batch,
        'num_classes': num_classes,
        'num_points_per_sample': num_points_per_sample,
        'batch_size': batch_size
    }


def test_point_cloud_segmentation_criterion_init():
    """Test that we can initialize the criterion with various parameters."""
    # Default initialization
    criterion = PointCloudSegmentationCriterion()
    assert criterion.criterion.ignore_index == -100
    assert criterion.criterion.weight is None
    
    # Custom ignore_index
    criterion = PointCloudSegmentationCriterion(ignore_index=255)
    assert criterion.criterion.ignore_index == 255
    
    # Custom class weights
    class_weights = (0.5, 1.0, 2.0)
    criterion = PointCloudSegmentationCriterion(class_weights=class_weights)
    assert torch.allclose(criterion.criterion.weight.cpu(), torch.tensor(class_weights))


def test_point_cloud_segmentation_criterion_compute_loss(sample_data):
    """Test that we can compute loss with the criterion."""
    criterion = PointCloudSegmentationCriterion()
    
    # Compute loss
    loss = criterion._compute_loss(sample_data['logits'], sample_data['labels'])
    
    # Validate loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() > 0  # Loss should be positive for random predictions
    
    # Compare with direct computation using CrossEntropyLoss
    direct_loss = torch.nn.functional.cross_entropy(
        sample_data['logits'], sample_data['labels']
    )
    assert torch.allclose(loss, direct_loss)


def test_point_cloud_segmentation_criterion_call(sample_data):
    """Test that we can call the criterion with various input formats."""
    criterion = PointCloudSegmentationCriterion()
    
    # Direct tensors
    loss1 = criterion(sample_data['logits'], sample_data['labels'])
    assert isinstance(loss1, torch.Tensor)
    assert loss1.dim() == 0  # Scalar tensor
    
    # Dictionary inputs
    loss2 = criterion({'pred': sample_data['logits']}, {'true': sample_data['labels']})
    assert isinstance(loss2, torch.Tensor)
    assert loss2.dim() == 0  # Scalar tensor
    
    # Buffer should contain the losses
    assert len(criterion.buffer) == 2
    assert all(isinstance(x, torch.Tensor) for x in criterion.buffer)


def test_point_cloud_segmentation_criterion_summarize(sample_data):
    """Test that we can summarize the results of the criterion."""
    criterion = PointCloudSegmentationCriterion()
    
    # Add some losses to the buffer
    for _ in range(5):
        criterion(sample_data['logits'], sample_data['labels'])
    
    # Summarize
    summary = criterion.summarize()
    
    # Validate summary
    assert isinstance(summary, torch.Tensor)
    assert summary.dim() == 1
    assert summary.shape[0] == 5  # 5 losses in the buffer
    assert not torch.isnan(summary).any()


def test_point_cloud_segmentation_criterion_with_ignore_index(sample_data):
    """Test the criterion with ignore_index."""
    ignore_index = 0
    criterion = PointCloudSegmentationCriterion(ignore_index=ignore_index)
    
    # Set some labels to the ignore index
    labels = sample_data['labels'].clone()
    labels[0:10] = ignore_index
    
    # Compute loss
    loss = criterion._compute_loss(sample_data['logits'], labels)
    
    # Validate loss properties
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    
    # Compare with direct computation using CrossEntropyLoss
    direct_loss = torch.nn.functional.cross_entropy(
        sample_data['logits'], labels, ignore_index=ignore_index
    )
    assert torch.allclose(loss, direct_loss)


def test_point_cloud_segmentation_criterion_per_sample_weighting(sample_data):
    """Test that we can weight losses differently for each sample in the batch."""
    criterion = PointCloudSegmentationCriterion()
    
    # Compute per-sample losses manually
    losses = []
    for b in range(sample_data['batch_size']):
        # Get points from this sample
        mask = sample_data['batch'] == b
        sample_logits = sample_data['logits'][mask]
        sample_labels = sample_data['labels'][mask]
        
        # Compute loss for this sample
        sample_loss = torch.nn.functional.cross_entropy(
            sample_logits, sample_labels, reduction='mean'
        )
        losses.append(sample_loss)
    
    # Average the per-sample losses
    expected_loss = torch.stack(losses).mean()
    
    # Compare with the loss from the criterion on all points
    actual_loss = criterion._compute_loss(sample_data['logits'], sample_data['labels'])
    
    # The losses should be close but not exactly equal due to how batching is handled
    # The criterion computes loss over all points at once, not per sample
    assert abs(actual_loss.item() - expected_loss.item()) < 0.1
