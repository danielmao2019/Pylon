from typing import List, Dict, Union
import torch
from criteria.vision_2d import SemanticSegmentationCriterion


class ChangeFormerCriterion(SemanticSegmentationCriterion):

    def __call__(self, y_pred: Union[torch.Tensor, List[torch.Tensor]], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        if isinstance(y_pred, torch.Tensor):
            y_pred = [y_pred]
        assert isinstance(y_pred, list) and all(isinstance(x, torch.Tensor) for x in y_pred)
        assert type(y_true) == dict and set(y_true.keys()) == {'change_map'}
        multi_scale_losses = torch.stack([
            self.criterion(x, y_true['change_map']) for x in y_pred
        ])
        total_loss = multi_scale_losses.mean()
        assert total_loss.ndim == 0, f"{total_loss.shape=}"
        self.buffer.append(total_loss.detach().cpu())
        return total_loss
