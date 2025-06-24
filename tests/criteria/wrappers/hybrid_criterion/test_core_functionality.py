import pytest
import torch
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion


def test_compute_loss_sum(hybrid_criterion, sample_tensor):
    """Test computing loss with sum reduction."""
    # Create a target tensor
    y_true = torch.randn_like(sample_tensor)

    # Compute loss using the wrapper
    loss = hybrid_criterion(y_pred=sample_tensor, y_true=y_true)

    # Compute expected loss (sum of both criteria)
    expected_loss = sum(
        torch.nn.MSELoss()(input=sample_tensor, target=y_true)
        for _ in range(len(hybrid_criterion.criteria))
    )

    # Check that the losses match
    assert loss.item() == expected_loss.item()

    # Check that loss is a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_compute_loss_mean(criteria_cfg, sample_tensor):
    """Test computing loss with mean reduction."""
    # Create a criterion with mean reduction
    criterion_mean = HybridCriterion(combine='mean', criteria_cfg=criteria_cfg)

    # Create a target tensor
    y_true = torch.randn_like(sample_tensor)

    # Compute loss using the wrapper
    loss = criterion_mean(y_pred=sample_tensor, y_true=y_true)

    # Compute expected loss (mean of both criteria)
    expected_loss = sum(
        torch.nn.MSELoss()(input=sample_tensor, target=y_true)
        for _ in range(len(criterion_mean.criteria))
    ) / len(criterion_mean.criteria)

    # Check that the losses match
    assert loss.item() == expected_loss.item()


def test_compute_loss_method_directly(hybrid_criterion, sample_tensor):
    """Test the _compute_loss method directly."""
    y_true = torch.randn_like(sample_tensor)
    
    # Call _compute_loss directly
    loss = hybrid_criterion._compute_loss(y_pred=sample_tensor, y_true=y_true)
    
    # Verify it returns a scalar tensor
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)


def test_loss_tensor_properties(hybrid_criterion, sample_tensor):
    """Test that losses have correct tensor properties."""
    y_true = torch.randn_like(sample_tensor)
    
    loss = hybrid_criterion(y_pred=sample_tensor, y_true=y_true)
    
    # Basic tensor properties
    assert isinstance(loss, torch.Tensor)
    assert loss.dtype == torch.float32
    assert loss.ndim == 0  # scalar
    
    # Numerical properties
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert not torch.isnan(loss), f"Loss is NaN: {loss}"
    assert not torch.isinf(loss), f"Loss is infinite: {loss}"
    assert loss.item() >= 0, f"Loss should be non-negative: {loss}"


def test_combine_options_exhaustive(dummy_criterion, sample_tensor):
    """Test all valid combine options."""
    criteria_cfg = [
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        },
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        }
    ]
    
    y_true = torch.randn_like(sample_tensor)
    
    # Test sum
    criterion_sum = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    loss_sum = criterion_sum(y_pred=sample_tensor, y_true=y_true)
    
    # Test mean
    criterion_mean = HybridCriterion(combine='mean', criteria_cfg=criteria_cfg)
    loss_mean = criterion_mean(y_pred=sample_tensor, y_true=y_true)
    
    # Mean should be half of sum for 2 identical criteria
    assert abs(loss_mean.item() - loss_sum.item() / 2) < 1e-6


def test_component_criterion_isolation(dummy_criterion, sample_tensor):
    """Test that component criteria are properly isolated."""
    criteria_cfg = [
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        },
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        }
    ]
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Verify each component criterion is independent
    assert len(criterion.criteria) == 2
    assert criterion.criteria[0] is not criterion.criteria[1]
    
    # Verify they don't share buffer state
    for component in criterion.criteria:
        assert component.use_buffer is False
        assert not hasattr(component, 'buffer')


def test_gradient_flow(dummy_criterion, sample_tensor):
    """Test that gradients flow properly through the hybrid criterion."""
    criteria_cfg = [
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        }
    ]
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Create tensors with gradients
    input_tensor = sample_tensor.clone().requires_grad_(True)
    target_tensor = torch.randn_like(sample_tensor)
    
    # Forward pass
    loss = criterion(y_pred=input_tensor, y_true=target_tensor)
    
    # Backward pass
    loss.backward()
    
    # Check that gradients were computed
    assert input_tensor.grad is not None
    assert not torch.allclose(input_tensor.grad, torch.zeros_like(input_tensor.grad))


def test_deterministic_computation(dummy_criterion):
    """Test that computation is deterministic with same inputs."""
    criteria_cfg = [
        {
            'class': PyTorchCriterionWrapper,
            'args': {'criterion': dummy_criterion}
        }
    ]
    
    criterion = HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
    
    # Use fixed seed for deterministic inputs
    torch.manual_seed(42)
    sample_input = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    sample_target = torch.randn(2, 3, 4, 4, dtype=torch.float32)
    
    # Compute losses multiple times
    loss1 = criterion(y_pred=sample_input, y_true=sample_target)
    loss2 = criterion(y_pred=sample_input, y_true=sample_target)
    
    # Should be identical
    assert torch.equal(loss1, loss2), f"Non-deterministic computation: {loss1} != {loss2}"
