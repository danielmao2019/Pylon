from typing import Dict, Any
import torch
from criteria.vision_3d.point_cloud_registration.buffer_criteria.desc_loss import ContrastiveLoss, cdist
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError
from metrics.wrappers.single_task_metric import SingleTaskMetric


class BUFFER_DescStageMetric(SingleTaskMetric):

    def __init__(self, **kwargs) -> None:
        super(BUFFER_DescStageMetric, self).__init__(**kwargs)
        self.desc_loss = ContrastiveLoss()
        self.isotropic_transform_error = IsotropicTransformError(use_buffer=False)

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor], idx: int) -> Dict[str, torch.Tensor]:
        assert isinstance(y_pred, dict), f"{type(y_pred)=}"
        assert y_pred.keys() == {
            'src_kpt', 'tgt_kpt', 'src_des', 'tgt_des', 'equi_score', 'gt_label',
        } | {
            'pose', 'src_axis', 'tgt_axis',
        }, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict), f"{type(y_true)=}"
        assert y_true.keys() == {'transform'}, f"{y_true.keys()=}"

        tgt_kpt, src_des, tgt_des = y_pred['tgt_kpt'], y_pred['src_des'], y_pred['tgt_des']
        _, _, accuracy = self.desc_loss(src_des, tgt_des, cdist(tgt_kpt, tgt_kpt))
        pre_label = torch.argmax(y_pred['equi_score'], dim=1)
        eqv_acc = (pre_label == y_pred['gt_label']).sum() / pre_label.shape[0]
        scores = {
            'desc_acc': accuracy,
            'eqv_acc': eqv_acc,
            **self.isotropic_transform_error(y_pred['pose'], y_true['transform'], idx),
        }
        self.add_to_buffer(scores, idx)
        return scores
