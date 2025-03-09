import pytest
import torch
from criteria.vision_3d import PointCloudSegmentationCriterion


@pytest.fixture
def sample_data():
    # Generate sample data
    num_points = 100
    num_classes = 5
    
    # Create random logits and labels
    logits = torch.randn(num_points, num_classes)
    labels = torch.randint(0, num_classes, (num_points,), dtype=torch.int64)
    
    return {
        'logits': logits,
        'labels': labels,
        'num_classes': num_classes
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
