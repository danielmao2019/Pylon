from typing import Tuple, Dict
import torch
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from criteria.vision_2d.dense_prediction.dense_classification.semantic_segmentation import SemanticSegmentationCriterion
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss
from utils.semantic_segmentation.one_hot_encoding import to_one_hot


class DsferNetCriterion(SingleTaskCriterion):

    def __init__(self, ignore_value: int = 255, lam: float = 1.0) -> None:
        super(DsferNetCriterion, self).__init__()
        self.pool1 = torch.nn.MaxPool2d(8, stride=8)
        self.pool2 = torch.nn.MaxPool2d(16, stride=16)
        self.ce_loss = SemanticSegmentationCriterion(ignore_value=ignore_value)
        self.criterion1 = DiceLoss(ignore_value=ignore_value)
        self.criterion2 = DiceLoss(ignore_value=ignore_value)
        self.lam = lam

    def __call__(
        self,
        y_pred: Tuple[torch.Tensor, ...],
        y_true: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        assert isinstance(y_pred, tuple) and len(y_pred) == 3
        assert all(isinstance(t, torch.Tensor) for t in y_pred)
        assert isinstance(y_true, dict)
        assert set(y_true.keys()) == {'change_map'}
        assert isinstance(y_true['change_map'], torch.Tensor)
        y_true = y_true['change_map']
        # Compute loss
        ce_loss = self.ce_loss(y_pred[0], y_true)
        label_4 = to_one_hot(self.pool1(y_true.unsqueeze(1).float()).long(), y_pred[0].size(1), self.ignore_value)
        label_5 = to_one_hot(self.pool2(y_true.unsqueeze(1).float()).long(), y_pred[0].size(1), self.ignore_value)
        consistent_loss1 = self.criterion1(y_pred[1], label_4)
        consistent_loss2 = self.criterion2(y_pred[2], label_5)
        loss = ce_loss + self.lam * 0.5 * (consistent_loss1 + consistent_loss2)
        assert loss.ndim == 0, f"{loss.shape=}"
        # Log loss
        self.buffer.append(loss.detach().cpu())
        return loss
