import pytest
from metrics.wrappers.hybrid_metric import HybridMetric


@pytest.fixture
def hybrid_metric(metrics_cfg):
    """Create a HybridMetric instance for testing."""
    return HybridMetric(metrics_cfg=metrics_cfg)
