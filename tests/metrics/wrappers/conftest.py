import pytest
import torch
from metrics.base_metric import BaseMetric


class DummyMetric(BaseMetric):
    """A dummy metric that returns simple scores for testing."""

    def __init__(self, metric_name: str = "test_metric", use_buffer: bool = True):
        super().__init__(use_buffer=use_buffer)
        self.metric_name = metric_name

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        # Simple dummy score computation
        score = torch.mean(torch.abs(y_pred - y_true))
        scores = {self.metric_name: score}
        self.add_to_buffer(scores)
        return scores

    def summarize(self, output_path=None):
        assert self.use_buffer and hasattr(self, 'buffer') and self.buffer is not None
        self._buffer_queue.join()
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0
        return {"aggregated": {self.metric_name: torch.tensor(0.5)}}


class AnotherDummyMetric(BaseMetric):
    """Another dummy metric for testing combinations."""

    def __init__(self, metric_name: str = "another_metric", use_buffer: bool = True):
        super().__init__(use_buffer=use_buffer)
        self.metric_name = metric_name

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> dict:
        # Different dummy score computation
        score = torch.mean((y_pred - y_true) ** 2)
        scores = {self.metric_name: score}
        self.add_to_buffer(scores)
        return scores

    def summarize(self, output_path=None):
        assert self.use_buffer and hasattr(self, 'buffer') and self.buffer is not None
        self._buffer_queue.join()
        assert self._buffer_queue.empty(), "Buffer queue is not empty when summarizing"
        assert len(self.buffer) != 0
        return {"aggregated": {self.metric_name: torch.tensor(0.3)}}


@pytest.fixture
def dummy_metric():
    """Fixture that provides a DummyMetric instance."""
    return DummyMetric(metric_name="test_metric")


@pytest.fixture
def another_dummy_metric():
    """Fixture that provides an AnotherDummyMetric instance."""
    return AnotherDummyMetric(metric_name="another_metric")


@pytest.fixture
def sample_tensor():
    """Create a sample tensor for testing."""
    return torch.randn(2, 3, 4, 4, dtype=torch.float32)


@pytest.fixture
def sample_target():
    """Create a sample target tensor for testing."""
    return torch.randn(2, 3, 4, 4, dtype=torch.float32)


@pytest.fixture
def metrics_cfg():
    """Create metric configs for testing."""
    return [
        {
            'class': DummyMetric,
            'args': {
                'metric_name': 'metric1',
            }
        },
        {
            'class': AnotherDummyMetric,
            'args': {
                'metric_name': 'metric2',
            }
        }
    ]