"""Integration tests for change detection criteria.

These criteria are copied from official repos, so we only test basic functionality,
not mathematical correctness.
"""
import pytest
import torch
from criteria.vision_2d.change_detection.dsamnet_criterion import DSAMNetCriterion
from criteria.vision_2d.change_detection.snunet_criterion import SNUNetCDCriterion
from criteria.vision_2d.change_detection.change_star_criterion import ChangeStarCriterion


@pytest.mark.parametrize("criterion_class,args,create_inputs", [
    (DSAMNetCriterion, {"dice_weight": 1.0}, "dsamnet"),
    (SNUNetCDCriterion, {}, "snunet"),
    (ChangeStarCriterion, {}, "changestar"),
])
def test_change_detection_criterion_basic_functionality(criterion_class, args, create_inputs):
    """Test basic functionality of change detection criteria."""
    # Initialize criterion
    criterion = criterion_class(**args)
    
    # Create dummy input data based on criterion type
    batch_size = 2
    num_classes = 2
    height, width = 32, 32
    
    if create_inputs == "dsamnet":
        # DSAMNet expects tuple of 3 tensors and dict with 'change_map'
        # prob: distance map, ds2/ds3: single channel outputs for dice loss
        y_pred = (
            torch.randn(batch_size, height, width, requires_grad=True),     # prob: distance map (3D)
            torch.randn(batch_size, 1, height, width, requires_grad=True),  # ds2: single channel (4D)
            torch.randn(batch_size, 1, height, width, requires_grad=True)   # ds3: single channel (4D)
        )
        y_true = {"change_map": torch.zeros(batch_size, height, width, dtype=torch.long)}
    elif create_inputs == "snunet":
        # SNUNet expects tensor and dict with 'change_map'
        y_pred = torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        y_true = {"change_map": torch.randint(0, num_classes, (batch_size, height, width))}
    elif create_inputs == "changestar":
        # ChangeStar expects dict of tensors and dict with 'change' and 'semantic'
        y_pred = {
            "change_12": torch.randn(batch_size, num_classes, height, width, requires_grad=True),
            "change_21": torch.randn(batch_size, num_classes, height, width, requires_grad=True),
            "semantic": torch.randn(batch_size, num_classes, height, width, requires_grad=True)
        }
        y_true = {
            "change": torch.randint(0, num_classes, (batch_size, height, width)),
            "semantic": torch.randint(0, num_classes, (batch_size, height, width))
        }
    
    # Test forward pass
    loss = criterion(y_pred=y_pred, y_true=y_true)
    
    # Basic checks
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Should be scalar
    assert loss.requires_grad
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Test backward pass
    loss.backward()


def test_dsamnet_criterion_with_auxiliary_outputs():
    """Test DSAMNet criterion with auxiliary outputs."""
    criterion = DSAMNetCriterion(dice_weight=1.0)
    
    batch_size = 2
    num_classes = 2
    height, width = 32, 32
    
    # DSAMNet expects tuple of 3 tensors and dict with 'change_map'
    y_pred = (
        torch.randn(batch_size, height, width, requires_grad=True),
        torch.randn(batch_size, 1, height, width, requires_grad=True),
        torch.randn(batch_size, 1, height, width, requires_grad=True)
    )
    y_true = {"change_map": torch.zeros(batch_size, height, width, dtype=torch.long)}
    
    loss = criterion(y_pred=y_pred, y_true=y_true)
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_change_detection_criterion_gpu():
    """Test change detection criteria on GPU."""
    criterion = DSAMNetCriterion(dice_weight=1.0).cuda()
    
    batch_size = 2
    num_classes = 2
    height, width = 32, 32
    
    # DSAMNet expects tuple of 3 tensors and dict with 'change_map'
    y_pred = (
        torch.randn(batch_size, height, width, device='cuda', requires_grad=True),
        torch.randn(batch_size, 1, height, width, device='cuda', requires_grad=True),
        torch.randn(batch_size, 1, height, width, device='cuda', requires_grad=True)
    )
    y_true = {"change_map": torch.zeros(batch_size, height, width, dtype=torch.long, device='cuda')}
    
    loss = criterion(y_pred=y_pred, y_true=y_true)
    
    assert loss.is_cuda
    assert not torch.isnan(loss)
    
    # Test backward pass on GPU
    loss.backward()


def test_change_detection_criterion_edge_cases():
    """Test edge cases for change detection criteria."""
    criterion = DSAMNetCriterion(dice_weight=1.0)
    
    # Test with single sample
    y_pred = (
        torch.randn(1, 16, 16, requires_grad=True),
        torch.randn(1, 1, 16, 16, requires_grad=True),
        torch.randn(1, 1, 16, 16, requires_grad=True)
    )
    y_true = {"change_map": torch.zeros(1, 16, 16, dtype=torch.long)}
    
    loss = criterion(y_pred=y_pred, y_true=y_true)
    assert isinstance(loss, torch.Tensor)
    
    # Test with different spatial dimensions
    y_pred = (
        torch.randn(2, 64, 128, requires_grad=True),
        torch.randn(2, 1, 64, 128, requires_grad=True),
        torch.randn(2, 1, 64, 128, requires_grad=True)
    )
    y_true = {"change_map": torch.zeros(2, 64, 128, dtype=torch.long)}
    
    loss = criterion(y_pred=y_pred, y_true=y_true)
    assert isinstance(loss, torch.Tensor)


def test_change_detection_criterion_summarize():
    """Test summarize functionality of change detection criteria."""
    criterion = DSAMNetCriterion(dice_weight=1.0)
    
    # Make multiple forward passes
    for _ in range(3):
        y_pred = (
            torch.randn(2, 32, 32, requires_grad=True),
            torch.randn(2, 1, 32, 32, requires_grad=True),
            torch.randn(2, 1, 32, 32, requires_grad=True)
        )
        y_true = {"change_map": torch.zeros(2, 32, 32, dtype=torch.long)}
        criterion(y_pred=y_pred, y_true=y_true)
    
    # Test summarize
    if hasattr(criterion, 'summarize'):
        result = criterion.summarize()
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 1  # Should be 1D tensor of losses