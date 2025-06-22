from typing import Dict, Any
import torch
from metrics.vision_3d.point_cloud_registration.isotropic_transform_error import IsotropicTransformError
from metrics.wrappers.single_task_metric import SingleTaskMetric


class BUFFER_RefStageMetric(SingleTaskMetric):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.isotropic_transform_error = IsotropicTransformError()

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert isinstance(y_pred, dict), f"{type(y_pred)=}"
        assert y_pred.keys() == {
            'src_ref', 'tgt_ref', 'src_s', 'tgt_s',
        } | {
            'pose', 'src_axis', 'tgt_axis',
        }, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict), f"{type(y_true)=}"
        assert y_true.keys() == {'transform'}, f"{y_true.keys()=}"

        src_axis, tgt_axis = y_pred['src_ref'], y_pred['tgt_ref']
        gt_trans = y_true['transform']
        src_axis = src_axis @ gt_trans[:3, :3].transpose(-1, -2)
        err = 1 - torch.cosine_similarity(src_axis, tgt_axis).abs()
        err = err.mean()
        scores = {
            'ref_error': err,
            **self.isotropic_transform_error(y_pred['pose'], y_true['transform']),
        }
        self.add_to_buffer(scores)
        return scores
