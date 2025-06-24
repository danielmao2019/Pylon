import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion


def test_call_with_list_input(aux_criterion, sample_tensors, sample_tensor):
    """Test calling the criterion with a list of predictions."""
    # Compute loss
    loss = aux_criterion(y_pred=sample_tensors, y_true=sample_tensor)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_call_with_dict_input(aux_criterion, sample_tensor_dict, sample_tensor):
    """Test calling the criterion with a dictionary of predictions."""
    # Compute loss
    loss = aux_criterion(y_pred=sample_tensor_dict, y_true=sample_tensor)

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_reduction_options(criterion_cfg, sample_tensors, sample_tensor):
    """Test different reduction options."""
    # Test with 'mean' reduction
    criterion_mean = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='mean')
    loss_mean = criterion_mean(y_pred=sample_tensors, y_true=sample_tensor)

    # Test with 'sum' reduction
    criterion_sum = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='sum')
    loss_sum = criterion_sum(y_pred=sample_tensors, y_true=sample_tensor)

    # The mean loss should be half of the sum loss
    assert abs(loss_mean.item() - loss_sum.item() / 2) < 1e-5


def test_loss_tensor_properties(aux_criterion, sample_tensors, sample_tensor):
    """Test that losses have correct tensor properties."""
    loss = aux_criterion(y_pred=sample_tensors, y_true=sample_tensor)

    # Basic tensor properties
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert loss.ndim == 0  # scalar

    # Numerical properties
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert not torch.isnan(loss), f"Loss is NaN: {loss}"
    assert not torch.isinf(loss), f"Loss is infinite: {loss}"
    assert loss.item() >= 0, f"Loss should be non-negative: {loss}"


def test_single_output_input(dummy_criterion, sample_tensor):
    """Test with single output (edge case)."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with single tensor in list
    single_output = [sample_tensor]
    target = torch.randn_like(sample_tensor)

    loss = criterion(y_pred=single_output, y_true=target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_many_outputs_input(dummy_criterion, sample_tensor):
    """Test with many outputs."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test with many tensors
    many_outputs = [sample_tensor.clone() for _ in range(5)]
    target = torch.randn_like(sample_tensor)

    loss = criterion(y_pred=many_outputs, y_true=target)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_dict_to_list_conversion(dummy_criterion, sample_tensor_dict, sample_tensor):
    """Test conversion from dict to list of predictions."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Test that dict input is properly converted to list
    loss_dict = criterion(y_pred=sample_tensor_dict, y_true=sample_tensor)

    # Compare with equivalent list input
    tensor_list = list(sample_tensor_dict.values())
    loss_list = criterion(y_pred=tensor_list, y_true=sample_tensor)

    # Should produce the same result
    assert torch.allclose(loss_dict, loss_list)


def test_reduction_mathematical_correctness(dummy_criterion, sample_tensors, sample_tensor):
    """Test that reduction options work mathematically correctly."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    # Test sum reduction
    criterion_sum = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='sum')
    loss_sum = criterion_sum(y_pred=sample_tensors, y_true=sample_tensor)

    # Test mean reduction
    criterion_mean = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg, reduction='mean')
    loss_mean = criterion_mean(y_pred=sample_tensors, y_true=sample_tensor)

    # Calculate expected individual losses
    individual_losses = []
    for tensor in sample_tensors:
        individual_loss = torch.nn.MSELoss()(tensor, sample_tensor)
        individual_losses.append(individual_loss)

    # Verify sum
    expected_sum = sum(individual_losses)
    assert torch.allclose(loss_sum, expected_sum, atol=1e-6)

    # Verify mean
    expected_mean = expected_sum / len(sample_tensors)
    assert torch.allclose(loss_mean, expected_mean, atol=1e-6)


def test_gradient_flow(dummy_criterion, sample_tensors, sample_tensor):
    """Test that gradients flow properly through the auxiliary outputs criterion."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Create tensors with gradients
    input_tensors = [t.clone().requires_grad_(True) for t in sample_tensors]
    target_tensor = torch.randn_like(sample_tensor)

    # Forward pass
    loss = criterion(y_pred=input_tensors, y_true=target_tensor)

    # Backward pass
    loss.backward()

    # Check that gradients were computed for all inputs
    for input_tensor in input_tensors:
        assert input_tensor.grad is not None
        assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))


def test_deterministic_computation(dummy_criterion, sample_tensors, sample_tensor):
    """Test that computation is deterministic with same inputs."""
    criterion_cfg = {
        'class': PyTorchCriterionWrapper,
        'args': {'criterion': dummy_criterion, 'use_buffer': False}
    }

    criterion = AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)

    # Compute losses multiple times
    loss1 = criterion(y_pred=sample_tensors, y_true=sample_tensor)
    loss2 = criterion(y_pred=sample_tensors, y_true=sample_tensor)

    # Should be identical
    assert torch.equal(loss1, loss2), f"Non-deterministic computation: {loss1} != {loss2}"
