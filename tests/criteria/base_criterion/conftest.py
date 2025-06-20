"""Shared fixtures for BaseCriterion tests."""
from typing import Optional
import pytest
import torch
from criteria.base_criterion import BaseCriterion


class ConcreteCriterion(BaseCriterion):
    """Concrete implementation of BaseCriterion for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.call_count = 0
        self.summarize_count = 0
    
    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        self.call_count += 1
        # Simulate loss computation
        loss = torch.tensor(self.call_count * 0.1, requires_grad=True)
        self.add_to_buffer(loss)
        return loss
    
    def summarize(self, output_path: Optional[str] = None) -> torch.Tensor:
        self.summarize_count += 1
        if not self.use_buffer:
            return torch.tensor([0.0])
        
        # Wait for buffer processing
        self._buffer_queue.join()
        
        if len(self.buffer) == 0:
            return torch.tensor([0.0])
        
        result = torch.stack(self.buffer, dim=0)
        if output_path:
            torch.save(result, output_path)
        return result


@pytest.fixture
def criterion_with_buffer():
    """Fixture providing ConcreteCriterion with buffer enabled."""
    return ConcreteCriterion()


@pytest.fixture
def criterion_without_buffer():
    """Fixture providing ConcreteCriterion with buffer disabled."""
    return ConcreteCriterion(use_buffer=False)


@pytest.fixture
def sample_tensors():
    """Fixture providing sample prediction and target tensors."""
    return {
        'y_pred': torch.randn(4, 10),
        'y_true': torch.randint(0, 10, (4,))
    }