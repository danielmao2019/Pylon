from typing import Dict, Any
import torch
from criteria.wrappers import SingleTaskCriterion


class BUFFER_RefStageCriterion(SingleTaskCriterion):

    def __call__(self, y_pred: Dict[str, Any], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(y_pred, dict)
        assert y_pred.keys() == {'src_ref', 'tgt_ref', 'src_s', 'tgt_s'}, f"{y_pred.keys()=}"
        assert isinstance(y_true, dict)
        assert y_true.keys() == {'transform'}, f"{y_true.keys()=}"

        src_axis, tgt_axis = y_pred['src_ref'], y_pred['tgt_ref']
        gt_trans = y_true['transform']
        src_axis = src_axis @ gt_trans[:3, :3].transpose(-1, -2)
        err = 1 - torch.cosine_similarity(src_axis, tgt_axis).abs()
        eps = (y_pred['src_s'][:, 0] + y_pred['tgt_s'][:, 0]) / 2
        ref_loss = (torch.log(eps) + err / eps).mean()
        self.add_to_buffer(ref_loss)
        return ref_loss
