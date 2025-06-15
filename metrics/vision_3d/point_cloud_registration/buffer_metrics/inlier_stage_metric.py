from typing import Dict, Any
import torch
from metrics.wrappers import SingleTaskMetric


class BUFFER_InlierStageMetric(SingleTaskMetric):

    def __init__(self, **kwargs) -> None:
        super(BUFFER_InlierStageMetric, self).__init__(**kwargs)
        self.L1_loss = torch.nn.L1Loss()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(y_pred, dict), f"{type(y_pred)=}"
        assert y_pred.keys() == {
            'pred_ind', 'gt_ind',
        } | {
            'pose', 'src_axis', 'tgt_axis',
        }, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict), f"{type(y_true)=}"
        assert y_true.keys() == {'transform'}, f"{y_true.keys()=}"

        pred_ind, gt_ind = y_pred['pred_ind'], y_pred['gt_ind']
        match_loss = self.L1_loss(pred_ind, gt_ind)
        scores = {
            'match_loss': match_loss,
        }
        self.add_to_buffer(scores)
        return scores
