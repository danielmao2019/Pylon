from typing import Tuple, Dict, Optional
import torch
from criteria.vision_2d import SemanticSegmentationCriterion, IoULoss, SSIMLoss
from criteria.wrappers import HybridCriterion, AuxiliaryOutputsCriterion


class FTNCriterion(HybridCriterion):

    def __init__(self, num_classes: Optional[int] = 2, **kwargs) -> None:
        # Set up criteria configurations for HybridCriterion
        criteria_cfg = [
            {
                'class': AuxiliaryOutputsCriterion,
                'args': {
                    'use_buffer': False,
                    'criterion_cfg': {
                        'class': SemanticSegmentationCriterion,
                        'args': {'use_buffer': False},
                    }
                }
            },
            {
                'class': AuxiliaryOutputsCriterion,
                'args': {
                    'use_buffer': False,
                    'criterion_cfg': {
                        'class': SSIMLoss,
                        'args': {'use_buffer': False},
                    }
                }
            },
            {
                'class': AuxiliaryOutputsCriterion,
                'args': {
                    'use_buffer': False,
                    'criterion_cfg': {
                        'class': IoULoss,
                        'args': {'use_buffer': False},
                    }
                }
            },
        ]

        super(FTNCriterion, self).__init__(
            combine='sum',
            criteria_cfg=criteria_cfg,
            **kwargs
        )

    def __call__(self, y_pred: Tuple[torch.Tensor, ...], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        # input checks
        assert isinstance(y_pred, tuple) and len(y_pred) == 4, f"{type(y_pred)=}, {len(y_pred)=}"
        assert all(isinstance(x, torch.Tensor) for x in y_pred)
        assert isinstance(y_true, dict) and set(y_true.keys()) == {'change_map'}

        total_loss = 0
        for criterion in self.criteria:
            total_loss += criterion(y_pred, y_true['change_map'])

        self.add_to_buffer(total_loss)
        return total_loss
