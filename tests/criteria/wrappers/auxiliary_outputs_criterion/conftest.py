import pytest
from criteria.wrappers.pytorch_criterion_wrapper import PyTorchCriterionWrapper
from criteria.wrappers.auxiliary_outputs_criterion import AuxiliaryOutputsCriterion


@pytest.fixture
def criterion_cfg(dummy_criterion):
    """Create a criterion config with a criterion that has registered buffers."""
    return {
        'class': PyTorchCriterionWrapper,
        'args': {
            'criterion': dummy_criterion,
            'use_buffer': False,
        },
    }


@pytest.fixture
def aux_criterion(criterion_cfg):
    """Create an AuxiliaryOutputsCriterion instance for testing."""
    return AuxiliaryOutputsCriterion(criterion_cfg=criterion_cfg)
