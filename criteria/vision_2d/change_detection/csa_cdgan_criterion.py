from typing import Dict
import torch
from criteria.wrappers import MultiTaskCriterion


class CSA_CDGAN_Criterion(MultiTaskCriterion):

    def __call__(self, y_pred: Dict[str, torch.Tensor], y_true: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # input checks
        assert isinstance(y_pred, dict) and set(y_pred.keys()) == {'gen_image', 'pred_real', 'pred_fake'}
        assert isinstance(y_true, dict) and set(y_true.keys()) == {'image'}
