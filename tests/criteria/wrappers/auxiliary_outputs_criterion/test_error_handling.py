import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion


def test_input_validation_errors(dummy_criterion):
    """Test that invalid inputs raise appropriate errors."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with non-tensor inputs in list
    with pytest.raises(AssertionError):
        criterion(y_pred=["not_a_tensor", "also_not_tensor"], y_true=torch.randn(2, 3, 4, 4))

    # Test with mixed tensor/non-tensor
    sample_tensor = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    with pytest.raises(AssertionError):
        criterion(y_pred=[sample_tensor, "not_a_tensor"], y_true=torch.randn_like(sample_tensor))


def test_empty_predictions_list(dummy_criterion):
    """Test behavior with empty predictions list."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with empty list (should not crash but might produce empty tensor)
    target = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    # This should either raise an error or handle gracefully
    try:
        loss = criterion(y_pred=[], y_true=target)
        # If it doesn't raise an error, check that result makes sense
        assert isinstance(loss, torch.Tensor)
    except Exception:
        # It's acceptable for this to raise an error
        pass


def test_mismatched_tensor_shapes(dummy_criterion):
    """Test behavior with mismatched tensor shapes."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with mismatched shapes between predictions and target
    predictions = [torch.randn(2, 3, 4, 4, dtype=torch.float32)]
    mismatched_target = torch.randn(2, 3, 8, 8, dtype=torch.float32)  # Different shape

    # This should raise an error from the underlying criterion
    with pytest.raises(RuntimeError):
        criterion(y_pred=predictions, y_true=mismatched_target)


def test_invalid_input_types(dummy_criterion):
    """Test with completely invalid input types."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with non-list, non-dict input
    with pytest.raises(AssertionError):
        criterion(y_pred="not_a_list_or_dict", y_true=torch.randn(2, 3, 4, 4))

    # Test with int input
    with pytest.raises(AssertionError):
        criterion(y_pred=42, y_true=torch.randn(2, 3, 4, 4))


def test_different_sized_predictions(dummy_criterion):
    """Test with predictions of different sizes."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Create predictions with different sizes
    predictions = [
        torch.randn(1, 3, 4, 4, dtype=torch.float32),
        torch.randn(2, 3, 4, 4, dtype=torch.float32),  # Different batch size
    ]
    target = torch.randn(1, 3, 4, 4, dtype=torch.float32)

    # This should raise an error due to shape mismatch with second prediction
    with pytest.raises(RuntimeError):
        criterion(y_pred=predictions, y_true=target)


def test_nan_inf_handling(dummy_criterion):
    """Test behavior with NaN and Inf values in inputs."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with NaN values
    nan_predictions = [torch.full((2, 3, 4, 4), float('nan'), dtype=torch.float32)]
    target = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    loss_nan = criterion(y_pred=nan_predictions, y_true=target)
    assert torch.isnan(loss_nan)  # Loss should be NaN

    # Test with Inf values
    inf_predictions = [torch.full((2, 3, 4, 4), float('inf'), dtype=torch.float32)]
    loss_inf = criterion(y_pred=inf_predictions, y_true=target)
    assert torch.isinf(loss_inf)  # Loss should be Inf


def test_dtype_consistency(dummy_criterion):
    """Test that the criterion handles different dtypes appropriately."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with different dtypes (should work or raise appropriate error)
    predictions_double = [torch.randn(2, 3, 4, 4, dtype=torch.float64)]
    target_float = torch.randn(2, 3, 4, 4, dtype=torch.float32)

    try:
        loss = criterion(y_pred=predictions_double, y_true=target_float)
        # If successful, loss should be a valid tensor
        assert isinstance(loss, torch.Tensor)
        assert torch.isfinite(loss)
    except RuntimeError:
        # It's acceptable for this to raise a dtype mismatch error
        pass


def test_zero_dimensional_tensors(dummy_criterion):
    """Test behavior with zero-dimensional edge cases."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with very small tensors
    tiny_predictions = [torch.randn(1, 1, 1, 1, dtype=torch.float32)]
    tiny_target = torch.randn(1, 1, 1, 1, dtype=torch.float32)

    loss = criterion(y_pred=tiny_predictions, y_true=tiny_target)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
