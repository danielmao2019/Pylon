from typing import Tuple, Dict, Union, Optional
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

    def __call__(
        self,
        y_pred: torch.Tensor,
        y_true: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        r"""
        Args:
            y_pred (torch.Tensor): a float32 tensor of shape (N, C, H, W) for predicted logits.
            y_true (torch.Tensor or Dict[str, torch.Tensor]): an int64 tensor of shape (N, H, W) for ground-truth mask.
                If a dictionary is provided, then it's length is assumed to be 1 and the only value is taken as ground truth.

        Returns:
            loss (torch.Tensor): a float32 scalar tensor for loss value.
        """
        # input checks
        if type(y_true) == dict:
            assert len(y_true) == 1, f"{y_true.keys()=}"
            y_true = list(y_true.values())[0]
        check_semantic_segmentation(y_pred=y_pred, y_true=y_true)
        # match resolution
        if y_pred.shape[-2:] != y_true.shape[-2:]:
            y_pred = torchvision.transforms.Resize(
                size=y_true.shape[-2:], interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST,
            )(y_pred)
        # compute loss
        loss = self.criterion(input=y_pred, target=y_true)
        assert loss.dim() == 0, f"{loss.shape=}"
        # log loss
        self.buffer.append(loss)
        return loss
