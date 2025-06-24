import pytest
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.hybrid_criterion import HybridCriterion


@pytest.fixture
def criteria_cfg(dummy_criterion):
    """Create criterion configs with criteria that have registered buffers."""
    return [
        {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': dummy_criterion,
            }
        },
        {
            'class': PyTorchCriterionWrapper,
            'args': {
                'criterion': dummy_criterion,
            }
        }
    ]


@pytest.fixture
def hybrid_criterion(criteria_cfg):
    """Create a HybridCriterion instance for testing."""
    return HybridCriterion(combine='sum', criteria_cfg=criteria_cfg)
