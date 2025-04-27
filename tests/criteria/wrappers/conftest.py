import pytest
import torch


class DummyCriterionWithBuffer(torch.nn.Module):
    """A dummy criterion with a registered buffer for testing."""
    def __init__(self):
        super().__init__()
        self.register_buffer('class_weights', torch.ones(10))
        self.use_buffer = True
        self.buffer = []

    def forward(self, y_pred, y_true):
        loss = torch.mean((y_pred - y_true) ** 2 * self.class_weights[0])
        if self.use_buffer:
            self.buffer.append(loss.detach().cpu())
        return loss

    def reset_buffer(self):
        self.buffer = []


@pytest.fixture
def dummy_criterion():
    """Fixture that provides a DummyCriterionWithBuffer instance."""
    return DummyCriterionWithBuffer()


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 4, 4)


@pytest.fixture
def sample_tensors():
    """Create multiple sample tensors for testing."""
    return [torch.randn(2, 3, 4, 4), torch.randn(2, 3, 4, 4)]


@pytest.fixture
def sample_tensor_dict():
    """Create a dictionary of sample tensors for testing."""
    return {
        'pred1': torch.randn(2, 3, 4, 4),
        'pred2': torch.randn(2, 3, 4, 4)
    }


@pytest.fixture
def sample_multi_task_tensors():
    """Create sample tensors for multi-task testing."""
    return {
        'task1': torch.randn(2, 3, 4, 4),
        'task2': torch.randn(2, 3, 4, 4)
    }
