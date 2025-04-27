from typing import Dict, Tuple, Union
import torch
from criteria.wrappers.single_task_criterion import SingleTaskCriterion
from criteria.vision_2d.dense_prediction.dense_classification.dice_loss import DiceLoss


class BatchBalancedContrastiveLoss:
    def __init__(self, margin: Union[int, float] = 2.0, ignore_value: int = 255) -> None:
        self.margin = margin
        self.ignore_value = ignore_value

    def __call__(self, distance: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        assert distance.ndim == label.ndim == 3
        assert distance.shape == label.shape

        pos_region = label == 1
        neg_region = label == 0
        pos_distance = torch.pow(distance, 2)
        neg_distance = torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        pos_num = torch.sum(pos_region.float()) + 0.0001
        neg_num = torch.sum(neg_region.float()) + 0.0001

        loss_1 = torch.sum(pos_region * pos_distance) / pos_num
        loss_2 = torch.sum(neg_region * neg_distance) / neg_num
        loss = loss_1 + loss_2
        assert loss.ndim == 0, f"{loss.shape=}"
        return loss


class DSAMNetCriterion(SingleTaskCriterion):

    def __init__(self, dice_weight: Union[int, float]) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.batch_balanced_contrastive_loss = BatchBalancedContrastiveLoss()
        self.dice_loss = DiceLoss()

    def __call__(self, y_pred: Tuple[torch.Tensor, ...], y_true: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert isinstance(y_pred, tuple) and len(y_pred) == 3, "y_pred must be a tuple of 3 tensors"
        assert all(isinstance(t, torch.Tensor) for t in y_pred), "y_pred must be a tuple of tensors"
        assert isinstance(y_true, dict), "y_true must be a dictionary"
        assert y_true.keys() == {'change_map'}, "dictionary keys of y_true must be exactly 'change_map'"

        # Extract tensors from y_pred
        prob, ds2, ds3 = y_pred
        target = y_true['change_map']

        # Compute losses
        dice_loss = 0.5 * (self.dice_loss(ds2, target) + self.dice_loss(ds3, target))
        contrastive_loss = self.batch_balanced_contrastive_loss(prob, target)

        # Combine losses
        total_loss = contrastive_loss + self.dice_weight * dice_loss
        self.add_to_buffer(total_loss)
        return total_loss
