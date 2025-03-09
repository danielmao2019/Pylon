import pytest
import torch
from criteria.vision_3d import PointCloudSegmentationCriterion


@pytest.fixture
def sample_data():
    # Generate sample data
    num_points = 100
    num_classes = 5
    batch_size = 2
    
    # Create random logits and labels (unbatched)
    logits_unbatched = torch.randn(num_points, num_classes)
    labels_unbatched = torch.randint(0, num_classes, (num_points,), dtype=torch.int64)
    
    # Create random logits and labels (batched)
    logits_batched = torch.randn(batch_size, num_points, num_classes)
    labels_batched = torch.randint(0, num_classes, (batch_size, num_points), dtype=torch.int64)
    
    return {
        'logits_unbatched': logits_unbatched,
        'labels_unbatched': labels_unbatched,
        'logits_batched': logits_batched,
        'labels_batched': labels_batched,
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


def test_point_cloud_segmentation_criterion_compute_loss_unbatched(sample_data):
    """Test that we can compute loss with unbatched inputs."""
    criterion = PointCloudSegmentationCriterion()
    
    # Compute loss with unbatched inputs
    loss = criterion._compute_loss(
        sample_data['logits_unbatched'], 
        sample_data['labels_unbatched']
    )
    
    # Validate loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() > 0  # Loss should be positive for random predictions
    
    # Compare with direct computation using CrossEntropyLoss
    direct_loss = torch.nn.functional.cross_entropy(
        sample_data['logits_unbatched'], sample_data['labels_unbatched']
    )
    assert torch.allclose(loss, direct_loss)


def test_point_cloud_segmentation_criterion_compute_loss_batched(sample_data):
    """Test that the criterion handles batched inputs correctly through SingleTaskCriterion."""
    criterion = PointCloudSegmentationCriterion()
    
    # Call with batched inputs
    loss = criterion(
        sample_data['logits_batched'], 
        sample_data['labels_batched']
    )
    
    # Validate loss properties
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # Scalar tensor
    assert not torch.isnan(loss)
    assert loss.item() > 0
    
    # The SingleTaskCriterion will handle the batched input by extracting just the first
    # batch element when using dictionaries
    assert len(criterion.buffer) == 1


def test_point_cloud_segmentation_criterion_call(sample_data):
    """Test that we can call the criterion with various input formats."""
    criterion = PointCloudSegmentationCriterion()
    
    # Direct tensors (unbatched)
    loss1 = criterion(sample_data['logits_unbatched'], sample_data['labels_unbatched'])
    assert isinstance(loss1, torch.Tensor)
    assert loss1.dim() == 0  # Scalar tensor
    
    # Dictionary inputs (unbatched)
    loss2 = criterion(
        {'pred': sample_data['logits_unbatched']}, 
        {'true': sample_data['labels_unbatched']}
    )
    assert isinstance(loss2, torch.Tensor)
    assert loss2.dim() == 0  # Scalar tensor
    
    # Direct tensors (batched)
    loss3 = criterion(sample_data['logits_batched'], sample_data['labels_batched'])
    assert isinstance(loss3, torch.Tensor)
    assert loss3.dim() == 0  # Scalar tensor
    
    # Dictionary inputs (batched)
    loss4 = criterion(
        {'pred': sample_data['logits_batched']},
        {'true': sample_data['labels_batched']}
    )
    assert isinstance(loss4, torch.Tensor)
    assert loss4.dim() == 0  # Scalar tensor
    
    # Buffer should contain the losses
    assert len(criterion.buffer) == 4
    assert all(isinstance(x, torch.Tensor) for x in criterion.buffer)


def test_point_cloud_segmentation_criterion_summarize(sample_data):
    """Test that we can summarize the results of the criterion."""
    criterion = PointCloudSegmentationCriterion()
    
    # Add some losses to the buffer
    for _ in range(3):
        criterion(sample_data['logits_unbatched'], sample_data['labels_unbatched'])
    
    for _ in range(2):
        criterion(sample_data['logits_batched'], sample_data['labels_batched'])
    
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
    
    # Set some labels to the ignore index (unbatched)
    labels_unbatched = sample_data['labels_unbatched'].clone()
    labels_unbatched[0:10] = ignore_index
    
    # Compute loss (unbatched)
    loss_unbatched = criterion._compute_loss(sample_data['logits_unbatched'], labels_unbatched)
    
    # Validate loss properties
    assert isinstance(loss_unbatched, torch.Tensor)
    assert not torch.isnan(loss_unbatched)
    
    # Set some labels to the ignore index (batched)
    labels_batched = sample_data['labels_batched'].clone()
    labels_batched[:, 0:10] = ignore_index
    
    # Call with batched inputs
    loss_batched = criterion(sample_data['logits_batched'], labels_batched)
    
    # Validate loss properties
    assert isinstance(loss_batched, torch.Tensor)
    assert not torch.isnan(loss_batched)
