from typing import Dict, Any
import torch
from metrics.wrappers import SingleTaskMetric


class BUFFER_InlierStageMetric(SingleTaskMetric):
    
    def __init__(self, **kwargs) -> None:
        super(BUFFER_InlierStageMetric, self).__init__(**kwargs)
        self.L1_loss = torch.nn.L1Loss()
    
    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        pred_ind, gt_ind = y_pred['pred_ind'], y_pred['gt_ind']
        match_loss = self.L1_loss(pred_ind, gt_ind)
        scores = {
            'match_loss': match_loss,
        }
        self.buffer.append(scores)
        return scores
