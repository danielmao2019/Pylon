from typing import Dict, Any
import torch
from criteria.vision_3d.point_cloud_registration.buffer_criteria.desc_loss import ContrastiveLoss, cdist
from criteria.wrappers import SingleTaskCriterion


class BUFFER_DescStageCriterion(SingleTaskCriterion):

    def __init__(self, **kwargs) -> None:
        super(BUFFER_DescStageCriterion, self).__init__(**kwargs)
        self.desc_loss = ContrastiveLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        tgt_kpt, src_des, tgt_des = y_pred['tgt_kpt'], y_pred['src_des'], y_pred['tgt_des']
        desc_loss, _, _ = self.desc_loss(src_des, tgt_des, cdist(tgt_kpt, tgt_kpt))
        eqv_loss = self.class_loss(y_pred['equi_score'], y_pred['gt_label'])
        loss = 4 * desc_loss + eqv_loss
        self.add_to_buffer(loss)
        return loss
