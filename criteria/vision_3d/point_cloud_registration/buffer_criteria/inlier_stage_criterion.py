from typing import Dict, Any
import torch
from criteria.wrappers import SingleTaskCriterion


class BUFFER_InlierStageCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs) -> None:
        super(BUFFER_InlierStageCriterion, self).__init__(**kwargs)
        self.L1_loss = torch.nn.L1Loss()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_ind, gt_ind = y_pred['pred_ind'], y_pred['gt_ind']
        match_loss = self.L1_loss(pred_ind, gt_ind)
        self.add_to_buffer(match_loss)
        return match_loss
