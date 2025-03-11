import pytest
import torch
from criteria.vision_3d.point_cloud_segmentation_criterion import PointCloudSegmentationCriterion


@pytest.mark.parametrize("batch_size, num_points, num_classes, ignore_index, class_weights", [
    (2, 1000, 4, None, None),  # Basic case
    (1, 500, 3, -1, (1.0, 2.0, 0.5)),  # With class weights
    (3, 2000, 5, 0, None),  # With ignore index
])
def test_point_cloud_segmentation_criterion(batch_size, num_points, num_classes, ignore_index, class_weights):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize criterion and move to device
    criterion = PointCloudSegmentationCriterion(
        ignore_index=ignore_index,
        class_weights=class_weights,
    ).to(device)
    
    # Create dummy predictions and targets
    y_pred = torch.randn(batch_size * num_points, num_classes).to(device)
    if ignore_index is not None:
        # Add some ignored points
        y_true = torch.randint(
            low=ignore_index, high=num_classes, 
            size=(batch_size * num_points,)
        ).to(device)
    else:
        y_true = torch.randint(
            low=0, high=num_classes,
            size=(batch_size * num_points,)
        ).to(device)

    # Compute loss
    loss = criterion(y_pred, y_true)
    
    # Basic checks
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # Scalar output
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0  # Cross entropy loss should be positive


@pytest.mark.parametrize("invalid_shape", [
    ((100, 5), (50,)),  # Different number of points
    ((100, 3), (100, 1)),  # Target has wrong shape
])
def test_invalid_shapes(invalid_shape):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = PointCloudSegmentationCriterion().to(device)
    
    y_pred = torch.randn(*invalid_shape[0]).to(device)
    y_true = torch.randint(0, 3, invalid_shape[1]).to(device)
    
    with pytest.raises(AssertionError):
        criterion(y_pred, y_true)


def test_class_weights():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    class_weights = (1.0, 2.0, 0.5)
    
    criterion = PointCloudSegmentationCriterion(
        class_weights=class_weights,
    ).to(device)
    
    # Verify that the weights are set correctly in the criterion
    assert criterion.criterion.weight is not None, "Class weights not set"
    
    # Convert raw weights to normalized weights that sum to 1
    raw_weights = torch.tensor(class_weights, dtype=torch.float32)
    normalized_weights = raw_weights / raw_weights.sum()
    
    assert torch.allclose(
        criterion.criterion.weight,
        normalized_weights.to(device)
    ), "Class weights not set correctly"
    
    # Create predictions and targets
    batch_size = 100
    y_pred = torch.randn(batch_size, num_classes).to(device)
    y_true = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # Verify that we can compute a loss without errors
    loss = criterion(y_pred, y_true)
    assert isinstance(loss, torch.Tensor)
    assert loss.numel() == 1
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0
