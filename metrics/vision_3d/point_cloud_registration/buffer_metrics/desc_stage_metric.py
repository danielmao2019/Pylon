from typing import Dict, Any
import torch
from metrics.wrappers import SingleTaskMetric
from criteria.vision_3d.point_cloud_registration.buffer_criteria.desc_loss import ContrastiveLoss, cdist
from utils.ops.apply import apply_tensor_op


class BUFFER_DescStageMetric(SingleTaskMetric):

    def __init__(self, **kwargs) -> None:
        super(BUFFER_DescStageMetric, self).__init__(**kwargs)
        self.desc_loss = ContrastiveLoss()
        self.class_loss = torch.nn.CrossEntropyLoss()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tgt_kpt, src_des, tgt_des = y_pred['tgt_kpt'], y_pred['src_des'], y_pred['tgt_des']
        _, _, accuracy = self.desc_loss(src_des, tgt_des, cdist(tgt_kpt, tgt_kpt))
        pre_label = torch.argmax(y_pred['equi_score'], dim=1)
        eqv_acc = (pre_label == y_pred['gt_label']).sum() / pre_label.shape[0]
        scores = {
            'desc_acc': accuracy,
            'eqv_acc': eqv_acc,
        }
        scores = apply_tensor_op(func=lambda x: x.detach().cpu(), inputs=scores)
        self.buffer.append(scores)
        return scores
