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
    
    # Initialize criterion
    criterion = PointCloudSegmentationCriterion(
        ignore_index=ignore_index,
        class_weights=class_weights,
        device=device
    )
    
    # Create dummy predictions and targets
    y_pred = torch.randn(batch_size * num_points, num_classes, device=device)
    if ignore_index is not None:
        # Add some ignored points
        y_true = torch.randint(
            low=ignore_index, high=num_classes, 
            size=(batch_size * num_points,), 
            device=device
        )
    else:
        y_true = torch.randint(
            low=0, high=num_classes,
            size=(batch_size * num_points,),
            device=device
        )

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
    criterion = PointCloudSegmentationCriterion(device=device)
    
    y_pred = torch.randn(*invalid_shape[0], device=device)
    y_true = torch.randint(0, 3, invalid_shape[1], device=device)
    
    with pytest.raises(AssertionError):
        criterion(y_pred, y_true)


def test_class_weights():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    class_weights = (1.0, 2.0, 0.5)
    
    criterion = PointCloudSegmentationCriterion(
        class_weights=class_weights,
        device=device
    )
    
    # Create predictions with equal probabilities after softmax
    y_pred = torch.zeros(100, num_classes, device=device)
    y_pred[:, 0] = 1.0  # exp(1) / (exp(1) + 1 + 1) ≈ 0.42
    y_pred[:, 1] = 1.0  # exp(1) / (exp(1) + 1 + 1) ≈ 0.42
    y_pred[:, 2] = 1.0  # exp(1) / (exp(1) + 1 + 1) ≈ 0.42
    
    # Create targets with different class distributions
    y_true_1 = torch.zeros(100, dtype=torch.long, device=device)  # All class 0 (weight 1.0)
    y_true_2 = torch.ones(100, dtype=torch.long, device=device)   # All class 1 (weight 2.0)
    
    # Loss for class 1 should be higher due to higher weight
    loss_1 = criterion(y_pred, y_true_1)
    loss_2 = criterion(y_pred, y_true_2)
    assert loss_2 > loss_1, "Loss with higher class weight should be larger"
