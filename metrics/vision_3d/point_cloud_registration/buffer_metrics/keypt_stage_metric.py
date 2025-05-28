from typing import Dict, Any
import torch
from metrics.wrappers import SingleTaskMetric
from criteria.vision_3d.point_cloud_registration.buffer_criteria.desc_loss import ContrastiveLoss, cdist


class BUFFER_KeyptStageMetric(SingleTaskMetric):
    
    def __init__(self, **kwargs) -> None:
        super(BUFFER_KeyptStageMetric, self).__init__(**kwargs)
        self.desc_loss = ContrastiveLoss()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        src_kpt, src_des, tgt_des = y_pred['src_kpt'], y_pred['src_des'], y_pred['tgt_des']
        _, _, accuracy = self.desc_loss(src_des, tgt_des, cdist(src_kpt, src_kpt))
        scores = {
            'desc_acc': accuracy,
        }
        self.buffer.append(scores)
        return scores
