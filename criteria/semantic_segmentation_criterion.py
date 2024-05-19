from typing import Tuple, Optional
import torch
import torchvision
from .base_criterion import BaseCriterion
from utils.input_checks import check_semantic_segmentation


class SemanticSegmentationCriterion(BaseCriterion):

    def __init__(self, ignore_index: int, weight: Optional[Tuple[float, ...]] = None) -> None:
        super(SemanticSegmentationCriterion, self).__init__()
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=weight, reduction='mean',
        )

    def _compute_loss_(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor): an int64 tensor of shape (N, H, W) for ground-truth mask.

        Returns:
            loss (torch.Tensor): a float32 scalar tensor for loss value.
        """
        # input checks
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)
        # compute loss
        return self.criterion(input=y_pred, target=y_true)
