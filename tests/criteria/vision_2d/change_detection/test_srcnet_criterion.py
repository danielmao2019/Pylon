import pytest
import torch
from criteria.vision_2d.change_detection.srcnet_criterion import SRCNetCriterion


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    height, width = 32, 32
    
    # Create prediction tensors
    prediction = torch.randn(batch_size, 2, height, width, device=device, requires_grad=True)  # 2 classes
    Dis = torch.randn(batch_size, 2, height, width, device=device, requires_grad=True)
    dif = torch.tensor(0.1, device=device, requires_grad=True)
    sigma = torch.tensor([1.0, 1.0, 1.0], device=device, requires_grad=True)  # 3 sigmas for the 3 losses
    
    # Create ground truth change map
    change_map = torch.randint(0, 2, (batch_size, height, width), device=device, requires_grad=False)
    
    return (prediction, Dis, dif, sigma), {'change_map': change_map}


def test_srcnet_criterion_basic(sample_data):
    """Test basic functionality of SRCNetCriterion with dummy data."""
    y_pred, y_true = sample_data
    criterion = SRCNetCriterion()
    
    # Compute loss
    loss = criterion(y_pred=y_pred, y_true=y_true)
    
    # Basic assertions
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert loss.item() > 0  # Loss should be positive
    assert loss.requires_grad  # Loss should require gradients
    
    # Test backward pass
    loss.backward()
    assert y_pred[0].grad is not None  # prediction should have gradients
    assert y_pred[1].grad is not None  # Dis should have gradients
    assert y_pred[2].grad is not None  # dif should have gradients
    assert y_pred[3].grad is not None  # sigma should have gradients
