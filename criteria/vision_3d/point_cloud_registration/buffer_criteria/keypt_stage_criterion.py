from typing import Dict, Any
import torch
from criteria.wrappers import SingleTaskCriterion
from criteria.vision_3d.point_cloud_registration.buffer_criteria.desc_loss import ContrastiveLoss, cdist


class BUFFER_KeyptStageCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs) -> None:
        super(BUFFER_KeyptStageCriterion, self).__init__(**kwargs)
        self.desc_loss = ContrastiveLoss()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        src_s, tgt_s = y_pred['src_s'], y_pred['tgt_s']
        src_kpt, src_des, tgt_des = y_pred['src_kpt'], y_pred['src_des'], y_pred['tgt_des']
        _, diff, _ = self.desc_loss(src_des, tgt_des, cdist(src_kpt, src_kpt))
        sigma = (src_s[:, 0] + tgt_s[:, 0]) / 2
        det_loss = torch.mean((1.0 - diff.detach()) * sigma)
        self.add_to_buffer(det_loss)
        return det_loss
