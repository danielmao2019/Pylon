"""Test learning rate lambda functions."""
import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from schedulers.lr_lambdas.warmup import WarmupLambda
from schedulers.lr_lambdas.constant import ConstantLambda


def test_warmup_lambda_initialization():
    """Test WarmupLambda initialization."""
    # Valid initialization
    lambda_fn = WarmupLambda(steps=100)
    assert lambda_fn.steps == 100
    
    # Edge case: minimum steps
    lambda_fn = WarmupLambda(steps=1)
    assert lambda_fn.steps == 1


def test_warmup_lambda_invalid_initialization():
    """Test WarmupLambda with invalid parameters."""
    # Invalid step count
    with pytest.raises(AssertionError):
        WarmupLambda(steps=0)
    
    with pytest.raises(AssertionError):
        WarmupLambda(steps=-10)
    
    # Invalid type
    with pytest.raises(AssertionError):
        WarmupLambda(steps=10.5)
    
    with pytest.raises(AssertionError):
        WarmupLambda(steps="100")


def test_warmup_lambda_progression():
    """Test warmup learning rate progression."""
    warmup_steps = 10
    lambda_fn = WarmupLambda(steps=warmup_steps)
    
    # Test progression during warmup
    assert lambda_fn(0) == 0.0  # Start at 0
    assert lambda_fn(1) == 0.1  # 1/10
    assert lambda_fn(5) == 0.5  # 5/10
    assert lambda_fn(9) == 0.9  # 9/10
    assert lambda_fn(10) == 1.0  # 10/10 - fully warmed up
    
    # Test after warmup
    assert lambda_fn(11) == 1.0  # Should stay at 1
    assert lambda_fn(100) == 1.0  # Should stay at 1


@pytest.mark.parametrize("steps,test_iter,expected", [
    (100, 0, 0.0),
    (100, 25, 0.25),
    (100, 50, 0.5),
    (100, 75, 0.75),
    (100, 100, 1.0),
    (100, 150, 1.0),
    (1, 0, 0.0),
    (1, 1, 1.0),
    (1, 2, 1.0),
])
def test_warmup_lambda_values(steps, test_iter, expected):
    """Test warmup values at specific iterations."""
    lambda_fn = WarmupLambda(steps=steps)
    assert lambda_fn(test_iter) == expected


def test_warmup_lambda_with_pytorch_scheduler():
    """Test WarmupLambda with PyTorch LambdaLR scheduler."""
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=1.0)
    
    # Create scheduler with warmup
    warmup_steps = 5
    lambda_fn = WarmupLambda(steps=warmup_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    
    # Test learning rate progression
    expected_lrs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0]
    
    for i, expected_lr in enumerate(expected_lrs):
        current_lr = optimizer.param_groups[0]['lr']
        assert abs(current_lr - expected_lr) < 1e-6
        scheduler.step()


def test_constant_lambda_initialization():
    """Test ConstantLambda initialization."""
    # Should accept any arguments
    lambda_fn = ConstantLambda()
    lambda_fn = ConstantLambda(123)
    lambda_fn = ConstantLambda(foo="bar", baz=42)
    
    # All should work fine
    assert callable(lambda_fn)


def test_constant_lambda_output():
    """Test ConstantLambda always returns 1."""
    lambda_fn = ConstantLambda()
    
    # Test various iterations
    for i in [0, 1, 10, 100, 1000, 10000]:
        assert lambda_fn(i) == 1


def test_constant_lambda_with_pytorch_scheduler():
    """Test ConstantLambda with PyTorch LambdaLR scheduler."""
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    base_lr = 0.1
    optimizer = SGD(model.parameters(), lr=base_lr)
    
    # Create scheduler with constant lambda
    lambda_fn = ConstantLambda()
    scheduler = LambdaLR(optimizer, lr_lambda=lambda_fn)
    
    # Learning rate should remain constant
    for i in range(10):
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr == base_lr
        scheduler.step()


def test_combined_lambdas_warmup_then_constant():
    """Test warmup followed by constant learning rate."""
    model = nn.Linear(10, 1)
    base_lr = 0.01
    optimizer = SGD(model.parameters(), lr=base_lr)
    
    # Create warmup lambda
    warmup_steps = 5
    warmup_lambda = WarmupLambda(steps=warmup_steps)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Test warmup phase
    for i in range(warmup_steps):
        expected_lr = base_lr * (i / warmup_steps)
        current_lr = optimizer.param_groups[0]['lr']
        assert abs(current_lr - expected_lr) < 1e-6
        scheduler.step()
    
    # Test constant phase after warmup
    for i in range(5):
        current_lr = optimizer.param_groups[0]['lr']
        assert current_lr == base_lr
        scheduler.step()


def test_combined_lambdas_multiple_parameter_groups():
    """Test schedulers with multiple parameter groups."""
    # Create model with multiple parameter groups
    model1 = nn.Linear(10, 5)
    model2 = nn.Linear(5, 1)
    
    optimizer = SGD([
        {'params': model1.parameters(), 'lr': 0.1},
        {'params': model2.parameters(), 'lr': 0.01}
    ])
    
    # Apply warmup to both groups
    warmup_lambda = WarmupLambda(steps=10)
    scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    
    # Check initial state
    assert optimizer.param_groups[0]['lr'] == 0.0
    assert optimizer.param_groups[1]['lr'] == 0.0
    
    # Step and check progression
    scheduler.step()  # Move to step 1
    assert abs(optimizer.param_groups[0]['lr'] - 0.01) < 1e-6  # 0.1 * 0.1
    assert abs(optimizer.param_groups[1]['lr'] - 0.001) < 1e-6  # 0.01 * 0.1