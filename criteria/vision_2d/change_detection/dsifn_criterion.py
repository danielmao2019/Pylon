from typing import Tuple, Dict
import torch
from criteria.wrappers import SingleTaskCriterion, SpatialPyTorchCriterionWrapper, AuxiliaryOutputsCriterion
from criteria.vision_2d import DiceLoss


class DSIFNCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs) -> None:
        super(DSIFNCriterion, self).__init__(**kwargs)
        self.bce_loss = AuxiliaryOutputsCriterion(SpatialPyTorchCriterionWrapper(torch.nn.BCEWithLogitsLoss()))
        self.dice_loss = AuxiliaryOutputsCriterion(DiceLoss())

    def __call__(self, y_pred: Tuple[torch.Tensor, ...], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        # input checks
        assert isinstance(y_pred, tuple)
        assert all(isinstance(x, torch.Tensor) for x in y_pred)
        assert isinstance(y_true, dict)
        assert set(y_true.keys()) == {'change_map'}
        # prepare y_true
        y_true = y_true['change_map']
        # compute losses
        bce_loss = self.bce_loss(y_pred, y_true.type(torch.float32).unsqueeze(1))
        dice_loss = self.dice_loss(tuple(map(lambda x: torch.cat([1-x, x], dim=1), y_pred)), y_true)
        total_loss = bce_loss + dice_loss
        self.add_to_buffer(total_loss)
        return total_loss
